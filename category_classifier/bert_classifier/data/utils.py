import json
import logging
import math
import os
import typing as T

import pandas as pd

from pcs_category_classifier.bert_classifier.config.constants import DataPartitioningConstants, GlobalConstants
from pcs_category_classifier.bert_classifier.project_setting import ProjectSettings
from pcs_category_classifier.utils import io_utils


logger = logging.getLogger(__name__)


def check_data_consistency_and_determine_num_labels_and_num_examples(
    data_dir_path: str, project_settings: ProjectSettings
) -> T.Tuple[int, int]:
    train_data_chunks = list_tokenized_data_chunks(data_dir_path)

    if train_data_chunks:
        logger.info("Reading the number of examples directly from the tokenized JSON file blocks.")
        num_labels, num_examples = determine_num_labels_and_num_examples_in_preprocessed_data(
            train_data_chunks, project_settings
        )
    else:
        logger.info("Pre-processed data not found - reading the number of examples from the repartitioned CSV file.")
        num_labels, num_examples = determine_num_labels_and_num_examples_in_a_csv_file(data_dir_path, project_settings)

    logger.info(f"Number of labels: {num_labels}, number of examples: {num_examples}.")
    return num_labels, num_examples


def list_tokenized_data_chunks(data_dir: str) -> T.List[str]:
    return [
        path
        for path in io_utils.list_filepaths(
            data_dir, filename_extension=DataPartitioningConstants.TOKENIZED_BLOCK_EXTENSION
        )
        if os.path.basename(path).startswith(DataPartitioningConstants.TOKENIZED_BLOCK_PREFIX)
    ]


def determine_num_labels_and_num_examples_in_a_csv_file(
    path: str, project_settings: ProjectSettings
) -> T.Tuple[int, int]:
    path_to_train_file = io_utils.get_datapath_to_repartitioned_csv(path)

    # todo: currently disabled until the moment we agree on the name of the "product/offer_id" column
    # todo: or any other type of test
    # perform_data_consistency_check(path_to_train_file, project_settings)

    labels_df = pd.read_csv(
        path_to_train_file,
        delimiter=GlobalConstants.CSV_DELIMITER,
        usecols=[project_settings.category_column_name],
        squeeze=True,
    )
    category_mapping_file_path = os.path.join(
        path, GlobalConstants.CATEGORY_MAPPING_DIR, GlobalConstants.CATEGORY_MAPPING_FILENAME
    )
    if io_utils.file_exists(category_mapping_file_path):
        num_labels = len(io_utils.load_json(category_mapping_file_path))
    else:
        num_labels = len(labels_df.unique())

    num_examples = len(labels_df)

    return num_labels, num_examples


def determine_num_labels_and_num_examples_in_preprocessed_data(
    train_data_chunks: T.List[str], project_settings: ProjectSettings
) -> T.Tuple[int, int]:
    labels = []
    num_examples = 0

    for chunk_file in train_data_chunks:
        chunk = io_utils.load_json(chunk_file)
        chunk_labels = [example[project_settings.label_field_name] for example in chunk]
        labels += chunk_labels
        num_examples += len(chunk)

    unique_labels = set(labels)
    return len(unique_labels), num_examples


def determine_num_steps_per_epoch(batch_size: int, world_size: int, num_examples: int) -> int:
    effective_batch_size = batch_size * world_size
    return math.ceil(num_examples / effective_batch_size)


def perform_data_consistency_check(path_to_train_file: str, project_settings: ProjectSettings) -> None:
    train_data_columns = sorted(
        pd.read_csv(
            path_to_train_file, delimiter=GlobalConstants.CSV_DELIMITER, nrows=0, index_col=[0]
        ).columns.tolist()
    )

    assert all(
        col_name in project_settings.expected_columns for col_name in train_data_columns
    ), f"Expected column names {project_settings.expected_columns}, found {train_data_columns}"

    logger.info("Data consistency check passed")
