import argparse
import logging
import math
import os
import typing as T
from uuid import UUID

import pandas as pd

from pcs_category_classifier.bert_classifier.config.constants import GlobalConstants
from pcs_category_classifier.bert_classifier.project_setting import ProjectSettings
from pcs_category_classifier.utils import data_utils, io_utils, tokenization_utils
from pcs_category_classifier.utils.logging_utils import configure_logging_for_gcp_training


logger = logging.getLogger(__name__)


def initialize_category_mapping(args: argparse.Namespace) -> T.Dict[str, int]:
    dataset_labels = load_dataset_labels(args)
    unique_labels = dataset_labels[args.category_column_name].unique()

    if args.category_mapping_file:
        category_mapping = io_utils.load_json(args.category_mapping_file)
    else:
        category_mapping = data_utils.generate_category_mapping_from_unique_categories(unique_labels)

        # we save it twice: in the data_dir (to make training easier) and job_dir (to make deployment easier)
        for destination_path in [args.input_data_file, args.job_dir]:
            io_utils.save_json(
                category_mapping,
                output_dir=os.path.join(destination_path, GlobalConstants.CATEGORY_MAPPING_DIR),
                output_filename=GlobalConstants.CATEGORY_MAPPING_FILENAME,
            )
    return category_mapping


def map_label(label: str) -> str:
    try:
        return str(int(label))
    except ValueError:
        try:
            return str(UUID(label))
        except ValueError:
            raise ValueError(f"Invalid label format, expected either int or UUID! Label: {label}")


def load_dataset_labels(args: argparse.Namespace) -> pd.DataFrame:
    path_to_data_file = io_utils.get_datapath_to_repartitioned_csv(args.input_data_file)
    input_df = pd.read_csv(
        path_to_data_file,
        delimiter=GlobalConstants.CSV_DELIMITER,
        usecols=[args.category_column_name],
        escapechar=GlobalConstants.CSV_ESCAPE_CHAR,
    ).dropna()
    input_df[args.category_column_name] = input_df[args.category_column_name].apply(lambda label: map_label(label))
    return input_df


def compute_number_of_examples(args: argparse.Namespace) -> int:
    dataset_labels = load_dataset_labels(args)
    return len(dataset_labels)


class PartitionJob:
    def __init__(self, project_setting: ProjectSettings) -> None:
        configure_logging_for_gcp_training()
        self._project_settings = project_setting
        self._parser = self._add_parser()

    def run(self):
        args = self._parser.parse_args()
        args.input_data_file = args.input_data_files[0]
        category_mapping = initialize_category_mapping(args)

        tokenizer = tokenization_utils.load_tokenizer(args.tokenizer_path)

        for input_file in args.input_data_files:
            args.input_data_file = input_file
            logger.info(f"Partitioning file {input_file}...")
            num_examples = compute_number_of_examples(args)
            logger.info(f"Total number of examples to tokenize: {num_examples}")
            chunk_len = math.ceil(num_examples / args.num_partitions)
            logger.info(f"Length of each partition will be: {chunk_len}")

            tokenization_utils.tokenize_text_and_map_category_ids_to_index(
                data_dir_path=args.input_data_file,
                tokenizer=tokenizer,
                output_entity_class=self._project_settings.entity,
                category_mapping=category_mapping,
                chunk_len=chunk_len,
                same_num_of_examples_per_chunk=True,
            )

    def _add_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()

        parser.add_argument("--job-dir", type=str, required=True)
        parser.add_argument(
            "--input-data-files",
            type=str,
            nargs="+",
            required=True,
            help="Absolute path to input CSV dataset to be processed and partitioned",
        )
        parser.add_argument(
            "--tokenizer-path",
            type=str,
            required=True,
            help="Absolute path to directory containing trained tokenizer files, i.e, merges.txt and vocab.json",
        )
        parser.add_argument(
            "--num-partitions",
            type=int,
            required=True,
            help="Target number of partitions, i.e number of chunks dataset should be partitioned to",
        )
        parser.add_argument(
            "--category-mapping-file",
            type=str,
            help="Absolute path to category_mapping.json used to map category_id to integer id."
            "If not specified, a new one will be generated from the input-data-file."
            "This arg should be set when partitioning a validation/test set using mapping already"
            "generated at training time",
        )
        parser.add_argument(
            "--entity",
            default=self._project_settings.entity,
            required=False,
            help=argparse.SUPPRESS,  # used only for internal coherence
        )
        parser.add_argument(
            "--category-column-name",
            default=self._project_settings.category_column_name,
            required=False,
            help=argparse.SUPPRESS,  # used only for internal coherence
        )
        return parser


if __name__ == "__main__":
    partition_job = PartitionJob()  # use your project setting as an argument to run it
    partition_job.run()
