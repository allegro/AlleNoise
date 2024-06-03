import argparse
import logging
import os
import typing as T
from abc import abstractmethod
from copy import deepcopy
from functools import reduce

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

from pcs_category_classifier.abstract_classifier.dataset import AbstractDataModule
from pcs_category_classifier.bert_classifier.config.constants import AnchorBertDatasetConstants, GlobalConstants
from pcs_category_classifier.bert_classifier.config.defaults import BertTrainingDefaults
from pcs_category_classifier.bert_classifier.project_setting import ProjectSettings
from pcs_category_classifier.utils import data_utils, io_utils, tokenization_utils


logger = logging.getLogger(__name__)


class BertDataset(Dataset):
    def __init__(self, data: T.List) -> None:
        self._data = data

    def __len__(self) -> int:
        return len(self._data)

    @abstractmethod
    def __getitem__(self, idx: int) -> T.Dict[str, T.Union[torch.Tensor, int]]:
        raise NotImplementedError
    

class AnchorDataset(BertDataset):
    def __getitem__(self, idx: int) -> T.Dict[str, T.Union[torch.Tensor, int]]:
        entity = self._data[idx]
        offer_id = entity[AnchorBertDatasetConstants.OFFER_ID_FILED_NAME]
        token_ids = entity[AnchorBertDatasetConstants.FEATURES_FIELD_NAME]
        label = entity[AnchorBertDatasetConstants.LABEL_FIELD_NAME]
        if AnchorBertDatasetConstants.TRUE_LABEL_FIELD_NAME in entity.keys():
            true_label = entity[AnchorBertDatasetConstants.TRUE_LABEL_FIELD_NAME]
        else:
            true_label = label
        return {
            AnchorBertDatasetConstants.OFFER_ID_FILED_NAME: offer_id,
            AnchorBertDatasetConstants.FEATURES_FIELD_NAME: torch.LongTensor(token_ids),
            AnchorBertDatasetConstants.LABEL_FIELD_NAME: label,
            AnchorBertDatasetConstants.TRUE_LABEL_FIELD_NAME: true_label,
            AnchorBertDatasetConstants.INDEX_FIELD_NAME: idx,
        }

    @property
    def token_ids(self) -> T.List[torch.Tensor]:
        return [entity[AnchorBertDatasetConstants.FEATURES_FIELD_NAME] for entity in self._data]

    @property
    def labels(self) -> T.List[int]:
        return [entity[AnchorBertDatasetConstants.LABEL_FIELD_NAME] for entity in self._data]


class BertDataModule(AbstractDataModule):
    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__(hparams)

        self._dataset_type = self._hparams.dataset_type
        self._category_column_name = self._hparams.category_column_name
        self._stage = self._hparams.stage
        self._category_mapping = self._initialize_category_mapping()
        self._tokenizer = tokenization_utils.load_tokenizer(self._hparams.tokenizer_path)
        self._train_dataset_complete = None

    def prepare_data(self) -> None:
        if self._stage == "test":
            tokenization_utils.tokenize_text_and_map_category_ids_to_index(
                data_dir_path=self._hparams.test_file_path,
                tokenizer=self._tokenizer,
                output_entity_class=self._hparams.entity,
                category_mapping=self._category_mapping,
            )
        elif self._stage == "train":
            tokenization_utils.tokenize_text_and_map_category_ids_to_index(
                data_dir_path=self._hparams.train_file_path,
                tokenizer=self._tokenizer,
                output_entity_class=self._hparams.entity,
                category_mapping=self._category_mapping,
            )

            tokenization_utils.tokenize_text_and_map_category_ids_to_index(
                data_dir_path=self._hparams.val_file_path,
                tokenizer=self._tokenizer,
                output_entity_class=self._hparams.entity,
                category_mapping=self._category_mapping,
            )
            if self._is_test_file_defined():
                tokenization_utils.tokenize_text_and_map_category_ids_to_index(
                    data_dir_path=self._hparams.test_file_path,
                    tokenizer=self._tokenizer,
                    output_entity_class=self._hparams.entity,
                    category_mapping=self._category_mapping,
                )

    def _is_test_file_defined(self) -> bool:
        return self._hparams.test_file_path != ""

    def setup(self, stage: str = None) -> None:
        logger.info("Building datasets")
        if stage == "test":
            self._test_dataset = self._build_dataset(self._hparams.test_file_path)
        elif stage == "fit":
            logger.info(f"Number of train labels: {self._hparams.num_labels}")
            logger.info(f"Number of train examples: {self._hparams.num_examples}")
            self._train_dataset = self._build_dataset(self._hparams.train_file_path)
            self._val_dataset = self._build_dataset(self._hparams.val_file_path)
            self._train_dataset_complete = deepcopy(self._train_dataset)

    @property
    def dataset_type(self):
        return self._dataset_type

    def _build_dataset(self, data_dir: str) -> BertDataset:
        """
        This method builds the dataset in two different ways, depending if we are in distributed mode or not.

        If we are in distributed mode, firstly fetch the current process `global_rank` and the total number
        of processes (a.k.a `world_size`); this will trigger loading different data partitions in different processes.
        If we are not in distributed mode, the whole partitions will be loaded on a single process.

        Disclaimer: to allow training in distributed setting, we are expecting that number of partitions will be a
        multiple of the number of processes.
        """

        data_chunks = io_utils.list_filepaths(data_dir, filename_extension=".json")
        data_chunks = [path for path in data_chunks if not path.endswith(GlobalConstants.CATEGORY_MAPPING_FILENAME)]
        logger.info(f"All data chunks: {data_chunks}")

        if self._hparams.train_distributed:
            # sort to preserve order among processes
            data_chunks.sort()

            process_global_rank = torch.distributed.get_rank()
            num_processes = torch.distributed.get_world_size()

            assert len(data_chunks) % num_processes == 0, "Number of partitions is not a multiple of num_processes"

            local_data_chunks = data_chunks[process_global_rank::num_processes]
        else:
            local_data_chunks = data_chunks

        logger.info(f"Loading {len(local_data_chunks)} data_chunks: {local_data_chunks}")

        dataset = reduce(list.__add__, [io_utils.load_json(data_chunk) for data_chunk in local_data_chunks])

        return self._dataset_type(dataset)

    def _initialize_category_mapping(self) -> T.Dict[str, int]:
        if self._stage == "test":
            category_mapping = io_utils.load_json(self._hparams.category_mapping_file_path)
        elif self._stage == "train":
            category_mapping = self._initialize_category_mapping_for_training()
        else:
            raise KeyError(
                f"stage: {self._stage} is not supported - " f"Please specify a supported stage. Choices: [train, test]"
            )
        return category_mapping

    def _initialize_category_mapping_for_training(self) -> T.Dict[str, int]:
        category_mapping_file_path = os.path.join(
            self._hparams.train_file_path,
            GlobalConstants.CATEGORY_MAPPING_DIR,
            GlobalConstants.CATEGORY_MAPPING_FILENAME,
        )
        if io_utils.file_exists(category_mapping_file_path):
            category_mapping = io_utils.load_json(category_mapping_file_path)
        else:
            category_mapping = self._generate_category_mapping(self._hparams.train_file_path)

            io_utils.save_json(
                category_mapping,
                os.path.join(self._hparams.train_file_path, GlobalConstants.CATEGORY_MAPPING_DIR),
                GlobalConstants.CATEGORY_MAPPING_FILENAME,
            )

        io_utils.save_json(
            category_mapping,
            os.path.join(self._hparams.job_dir, GlobalConstants.CATEGORY_MAPPING_DIR),
            GlobalConstants.CATEGORY_MAPPING_FILENAME,
        )
        return category_mapping

    def _generate_category_mapping(self, path: str) -> T.Dict[str, int]:
        path_to_train_file = io_utils.get_datapath_to_repartitioned_csv(path)
        categories = pd.read_csv(
            path_to_train_file,
            delimiter=GlobalConstants.CSV_DELIMITER,
            usecols=[self._category_column_name],
            squeeze=True,
        ).unique()
        return data_utils.generate_category_mapping_from_unique_categories(categories)

    def _set_train_subset(self, selected_idx: T.List[int]) -> None:
        """Override the train dataset so that it contains only data points with the selected indices."""
        subset = list(map(self._train_dataset_complete._data.__getitem__, selected_idx))
        self._train_dataset = self._dataset_type(subset)

    @staticmethod
    def add_argparse_args(
        parent_parser: argparse.ArgumentParser, project_settings: ProjectSettings
    ) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--train-file-path", type=str, default="")
        parser.add_argument("--val-file-path", type=str, default="")
        parser.add_argument("--test-file-path", type=str, default="")
        parser.add_argument("--tokenizer-path", type=str, required=True)
        parser.add_argument(
            "--dataset-type",
            default=project_settings.dataset,
            required=False,
            help=argparse.SUPPRESS,  # used only for internal coherence
        )
        parser.add_argument(
            "--category-column-name",
            default=project_settings.category_column_name,
            required=False,
            help=argparse.SUPPRESS,  # used only for internal coherence
        )
        parser.add_argument("--batch-size", default=BertTrainingDefaults.BATCH_SIZE, type=int)
        parser.add_argument("--num-workers", default=BertTrainingDefaults.NUM_WORKERS, type=int)

        return parser
