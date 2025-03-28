import json
import typing as T
from io import TextIOWrapper

import fsspec


def choose_file_system(path: str) -> fsspec.AbstractFileSystem:
    if path.startswith("gs://"):
        return fsspec.filesystem("gcs")
    else:
        return fsspec.filesystem("file")


def open_file(path: str, mode="r") -> TextIOWrapper:
    fs = choose_file_system(path)
    return fs.open(path, mode)


def isdir(path: str) -> bool:
    fs = choose_file_system(path)
    return fs.isdir(path)


def makedirs(path: str) -> None:
    fs = choose_file_system(path)
    fs.makedirs(path, exist_ok=True)


def file_exists(path: str) -> bool:
    fs = choose_file_system(path)
    return fs.exists(path)


def list_filepaths(
    data_path: str,
    check_filename_extension: bool = True,
    filename_extension: str = ".csv",
) -> T.List[str]:
    fs = choose_file_system(data_path)
    file_paths = fs.ls(data_path)
    if check_filename_extension:
        file_paths = [file_path for file_path in file_paths if file_path.endswith(filename_extension)]
    if "gcs" in fs.protocol:
        return ["gs://" + file_path for file_path in file_paths]
    else:
        return file_paths


def load_json(path: str) -> T.Dict:
    with open_file(path, "r") as f:
        return json.load(f)