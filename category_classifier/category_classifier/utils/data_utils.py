import csv
import ctypes as ct
import os
import random
import typing as T
from concurrent.futures import ProcessPoolExecutor
from csv import DictReader
from itertools import chain

import numpy as np
import pandas as pd
import torch

from category_classifier.utils.utils_constants import GlobalUtilsConstants


CsvChunk = T.List[T.Dict[str, str]]


def set_csv_field_size_limit() -> None:
    """
    Done in order to avoid `_csv.Error: field larger than field limit (131072)`
    see: https://stackoverflow.com/questions/15063936/csv-error-field-larger-than-field-limit-131072
    """
    csv.field_size_limit(int(ct.c_ulong(-1).value // 2))


def get_worker_count() -> int:
    return os.cpu_count() - 2


def split_list_into_chunks(data_list: T.List[T.Any], num_chunks: int) -> T.List[T.List]:
    split_size = len(data_list) // num_chunks
    remainder = data_list[num_chunks * split_size :]
    data_split = [data_list[i * split_size : (i + 1) * split_size] for i in range(0, num_chunks)]
    data_split[-1].extend(remainder)
    return data_split


def readlines_generator(file_object: DictReader, chunk_len: int = 1000000) -> T.Iterator[CsvChunk]:
    """Lazy function (generator) to read a CSV file in line blocks.
    A DictReader object is expected as file_object reader.
    """
    while True:
        data = []
        for _ in range(chunk_len):
            try:
                line = next(file_object)
            except StopIteration:
                break
            data.append(line)
        if not data:
            break
        yield data


def parallelize_data_block(
    executor_pool: ProcessPoolExecutor,
    block_content: T.List[T.Any],
    transformation_fn: T.Any,
    num_workers: int,
) -> T.List[T.Any]:
    """
    Apply `transformation_fn` in a parallel fashion
    """
    data_chunks = split_list_into_chunks(block_content, num_workers)
    future = executor_pool.map(transformation_fn, data_chunks)
    return list(chain.from_iterable(future))


def pandas_to_numpy(dataframe: pd.DataFrame) -> np.ndarray:
    numpy_array = np.asarray(dataframe.to_list())
    numpy_array = numpy_array.reshape(numpy_array.shape[0], 1)
    return numpy_array


def generate_category_mapping_from_unique_categories(categories_df: pd.DataFrame) -> T.Dict[str, int]:
    return {str(categories_df[i]): i for i in range(len(categories_df))}


def seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (GlobalUtilsConstants.MAX_SEED_VALUE + 1)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
