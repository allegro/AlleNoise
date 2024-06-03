import csv
import logging
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from typing import Any, Dict, Iterable, List

from transformers import PreTrainedTokenizerFast

from pcs_category_classifier.bert_classifier.config.constants import (
    DataPartitioningConstants,
    GlobalConstants,
    TokenizerConstants,
)
from pcs_category_classifier.bert_classifier.data.entities import CategoryClassificationEntity
from pcs_category_classifier.bert_classifier.data.utils import list_tokenized_data_chunks
from pcs_category_classifier.utils import data_utils, io_utils


logger = logging.getLogger(__name__)


def tokenize_text_and_map_category_ids_to_index(
    data_dir_path: str,
    tokenizer: PreTrainedTokenizerFast,
    output_entity_class: CategoryClassificationEntity,
    category_mapping: Dict[str, int],
    chunk_len: int = 1000000,
    same_num_of_examples_per_chunk: bool = False,
) -> None:

    train_data_chunks = list_tokenized_data_chunks(data_dir_path)

    if not train_data_chunks:
        logger.info(
            "Pre-processed data not found - starting parallel tokenization and mapping process\n"
            f"Starting ProcessPoolExecutor with {data_utils.get_worker_count()} processes"
        )
        function_to_run_in_parallel = partial(load_item_from_csv, output_entity_class, tokenizer, category_mapping)
        with ProcessPoolExecutor(max_workers=data_utils.get_worker_count()) as executor_pool:
            tokenize_and_map(
                executor=executor_pool,
                function_to_run_in_parallel=function_to_run_in_parallel,
                data_dir=data_dir_path,
                chunk_len=chunk_len,
                same_num_of_examples_per_chunk=same_num_of_examples_per_chunk,
            )


def tokenize_and_map(
    executor: ProcessPoolExecutor,
    function_to_run_in_parallel: partial,
    data_dir: str,
    chunk_len: int,
    same_num_of_examples_per_chunk: bool,
) -> None:

    data_utils.set_csv_field_size_limit()
    data_file_to_be_chunked = io_utils.get_datapath_to_repartitioned_csv(data_dir)
    with io_utils.open_file(data_file_to_be_chunked, "r") as file_handle:
        for block_index, block_content in enumerate(
            data_utils.readlines_generator(
                csv.DictReader(
                    file_handle, delimiter=GlobalConstants.CSV_DELIMITER, escapechar=GlobalConstants.CSV_ESCAPE_CHAR
                ),
                chunk_len=chunk_len,
            )
        ):
            logger.info(f"Processing entities file {data_file_to_be_chunked} block {block_index}")

            tokenized_block = data_utils.parallelize_data_block(
                executor, block_content, function_to_run_in_parallel, data_utils.get_worker_count()
            )

            if _should_block_be_augmented(same_num_of_examples_per_chunk, len(tokenized_block), chunk_len):
                logger.info(f"augmenting block {block_index}")
                augment_block(tokenized_block, chunk_len)

            logger.info(
                f"file {data_file_to_be_chunked} block {block_index} processing finished,"
                f"created: {len(tokenized_block)} examples"
            )

            output_file_name = (
                f"{DataPartitioningConstants.TOKENIZED_BLOCK_PREFIX}"
                f"_{block_index}"
                f"{DataPartitioningConstants.TOKENIZED_BLOCK_EXTENSION}"
            )
            io_utils.save_json(tokenized_block, data_dir, output_file_name)


def load_item_from_csv(
    entity_class: CategoryClassificationEntity,
    tokenizer: PreTrainedTokenizerFast,
    category_mapping: Dict[str, int],
    csv_entities: Iterable[str],
) -> List[List[int]]:
    mapped_entities = map(entity_class.from_csv_entity, csv_entities)
    filtered_entities = filter(lambda entity: entity.is_correct(), mapped_entities)

    return list(
        map(
            lambda entity: entity.get_transformed_entity(tokenizer=tokenizer, category_mapping=category_mapping),
            filtered_entities,
        )
    )


def _should_block_be_augmented(
    same_num_of_examples_per_chunk: bool, tokenized_block_length: int, chunk_length: int
) -> bool:
    return same_num_of_examples_per_chunk and tokenized_block_length != chunk_length


def augment_block(tokenized_block: List[Any], chunk_len: int) -> List[Any]:
    logger.info(f"original block length: {len(tokenized_block)}")
    num_missing_examples = chunk_len - len(tokenized_block)
    logger.info(f"number of examples to add: {num_missing_examples}")
    repeated_examples = tokenized_block[-num_missing_examples:]
    tokenized_block.extend(repeated_examples)
    return tokenized_block


def load_tokenizer(tokenizer_path: str) -> PreTrainedTokenizerFast:
    tokenizer_dir = io_utils.translate_gcs_path_to_local(tokenizer_path)
    tokenizer_file = os.path.join(tokenizer_dir, TokenizerConstants.TOKENIZER_FILENAME)
    return PreTrainedTokenizerFast(tokenizer_file=tokenizer_file, pad_token=TokenizerConstants.PAD_TOKEN)
