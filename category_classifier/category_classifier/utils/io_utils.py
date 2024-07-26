import json
import logging
import os
import pickle
import typing as T
from io import BytesIO, TextIOWrapper

import fsspec
import torch
import yaml


logger = logging.getLogger(__name__)


def choose_file_system(path: str) -> fsspec.AbstractFileSystem:
    if path.startswith("gs://"):
        return fsspec.filesystem("gcs")
    else:
        return fsspec.filesystem("file")


def open_file(path: str, mode="r") -> TextIOWrapper:
    fs = choose_file_system(path)
    return fs.open(path, mode)


def load_pickle(path: str) -> T.Any:
    logger.info(f"Loading {path}")
    with open_file(path, mode="rb") as f_:
        return pickle.load(f_)


def load_json(path: str) -> T.Dict[T.Any, T.Any]:
    with open_file(path) as f_:
        return json.load(f_)


def load_yaml(path: str) -> T.Any:
    logger.info(f"Loading {path}")
    with open_file(path) as f:
        return yaml.load(f, Loader=yaml.SafeLoader)


def save_pickle(obj: T.Any, output_dir: str, output_filename: str) -> None:
    output_path = os.path.join(output_dir, output_filename)
    logger.info(f"Saving {output_path}")
    if not isdir(output_dir):
        makedirs(output_dir)
    with open_file(output_path, mode="wb") as f_:
        pickle.dump(obj, f_, protocol=5)


def save_json(obj: T.Any, output_dir: str, output_filename: str) -> None:
    output_path = os.path.join(output_dir, output_filename)
    logger.info(f"Saving {output_path}")
    if not isdir(output_dir):
        makedirs(output_dir)
    with open_file(output_path, mode="w") as f_:
        json.dump(obj, f_)


def isdir(path: str) -> bool:
    fs = choose_file_system(path)
    return fs.isdir(path)


def makedirs(path: str) -> None:
    fs = choose_file_system(path)
    fs.makedirs(path, exist_ok=True)


def file_exists(path: str) -> bool:
    fs = choose_file_system(path)
    return fs.exists(path)


def get_datapath_to_repartitioned_csv(path: str) -> T.Any:
    """
    This method is used to fetch the data path pointing to a CSV file that was generated
    by repartitioning a Spark Dataframe to a unique .CSV file
    """
    logger.info(f"Loading CSV files from {path}")
    all_csv_files = list_filepaths(path, filename_extension=".csv")
    if len(all_csv_files) > 1:
        logger.warning("Only one CSV file was expected but we found more. Loading only the first one in the list")
    return all_csv_files[0]


def list_filepaths(
    data_path: str, check_filename_extension: bool = True, filename_extension: str = ".csv"
) -> T.List[str]:
    fs = choose_file_system(data_path)
    file_paths = fs.ls(data_path)
    if check_filename_extension:
        file_paths = [file_path for file_path in file_paths if file_path.endswith(filename_extension)]
    if "gcs" in fs.protocol:
        return ["gs://" + file_path for file_path in file_paths]
    else:
        return file_paths


def translate_gcs_path_to_local(path: str) -> str:
    if path.startswith("gs://"):
        path = path.rstrip("/")
        local_path = os.path.join("/tmp", os.path.split(path)[-1])
        copy_dir(path, local_path)
        return local_path
    return path


def copy_dir(source_dir: str, target_dir: str) -> str:
    """
    - Creates the target_dir
    - Copies all files from the source_dir to the target_dir
    - Caution: it's not recursive
    - Works for both local directories and GCS
    """
    if not isdir(target_dir):
        makedirs(target_dir)
    source_files = list_filepaths(source_dir, check_filename_extension=False)
    for source_file in source_files:
        target_file = os.path.join(target_dir, os.path.basename(source_file))
        with open_file(source_file, mode="rb") as source, open_file(target_file, mode="wb") as target:
            content = source.read()
            target.write(content)


def torch_save_state_dict(to_save: T.Any, filename: str) -> None:
    with BytesIO() as bytes_buf:
        torch.save(to_save, bytes_buf)
        with open_file(filename, mode="wb") as f_:
            f_.write(bytes_buf.getvalue())


def torch_load_state_dict(path: str, *args, **kwargs) -> T.Any:
    return torch.load(open_file(path, "rb"), *args, **kwargs)


def is_cuda_available() -> bool:
    return torch.cuda.is_available()


def detect_device() -> str:
    return "cuda" if is_cuda_available() else "cpu"
