import typing as T
from collections import OrderedDict
from io import TextIOWrapper

import fsspec
from pyspark.sql import Row
from pyspark.sql.dataframe import DataFrame

from infrastructure.spark_utils import SparkSession


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


def save_dict_to_json(args: T.Dict[str, str], spark: SparkSession, output_dir: str) -> None:
    non_empty_items = {k: v for k, v in args.items() if v is not None}
    row = Row(**OrderedDict(non_empty_items))

    df = spark.createDataFrame([row])
    (df.coalesce(1).write.format("json").mode("overwrite").save(output_dir))


def save_spark_df_to_csv(df: DataFrame, file_path: str) -> None:
    (
        df.repartition(1).write.csv(
            file_path,
            compression="none",
            header=True,
            mode="overwrite",
            sep="\t",
        )
    )


def load_spark_df_from_csv(spark: SparkSession, csv_dir: str) -> DataFrame:
    assert isdir(csv_dir)
    return spark.read.csv(csv_dir, sep="\t", header=True)
