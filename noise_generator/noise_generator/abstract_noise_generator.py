from abc import ABC
from typing import List

from noise_generator.data_model import DataPoint, PerturbedDataPoint


class AbstractNoiseGenerator(ABC):
    def __init__(self):
        super().__init__()

    @staticmethod
    def identity_perturbance(data_point: DataPoint) -> PerturbedDataPoint:
        return PerturbedDataPoint(
            offer_id=data_point.offer_id,
            text=data_point.text,
            category_id=data_point.category_id,
            original_category_id=data_point.category_id,
        )

    @staticmethod
    def label_perturbance(data_point: DataPoint, noisy_label: str) -> PerturbedDataPoint:
        return PerturbedDataPoint(
            offer_id=data_point.offer_id,
            text=data_point.text,
            category_id=noisy_label,
            original_category_id=data_point.category_id,
        )

    @staticmethod
    def _get_dataset_categories(dataset: List[DataPoint]) -> List[str]:
        return list(map(lambda data_point: data_point.category_id, dataset))

    def _get_unique_categories(self, dataset: List[DataPoint]) -> List[str]:
        return list(set(self._get_dataset_categories(dataset)))
