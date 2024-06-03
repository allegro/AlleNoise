import logging
from typing import List, Tuple

import pandas as pd

from noise_generator.data_model import DataPoint, PerturbedDataPoint


logger = logging.getLogger(__name__)


def map_dataset_row_to_datapoint(row: Tuple) -> DataPoint:
    return DataPoint(offer_id=str(row[0]), text=str(row[1]), category_id=str(row[2]))


def load_dataset(dataset_path: str) -> List[DataPoint]:
    logger.info(f"Loading the dataset from file {dataset_path}...")
    dataset_df = pd.read_csv(dataset_path, sep="\t", escapechar="\\")
    datapoints = list(map(map_dataset_row_to_datapoint, dataset_df.values.tolist()))
    logger.info(f"Loaded {len(datapoints)} data points.")
    return datapoints


def save_perturbed_dataset(perturbed_dataset: List[PerturbedDataPoint], output_path: str) -> None:
    logger.info(f"Saving the perturbed dataset to a dataframe...")
    df = pd.DataFrame(
        {
            "offer_id": [datapoint.offer_id for datapoint in perturbed_dataset],
            "text": [datapoint.text for datapoint in perturbed_dataset],
            "category_id": [datapoint.category_id for datapoint in perturbed_dataset],
            "category_id_true": [datapoint.original_category_id for datapoint in perturbed_dataset],
        }
    )

    logger.info(f"Saving the dataframe to file {output_path}...")
    df.to_csv(output_path, sep="\t", escapechar="\\", index=False)
    logger.info("Done.")
