from abc import ABC, abstractmethod
from typing import List

from noise_generator.data_model import DataPoint, NoiseType, PerturbedDataPoint


class NoiseGeneratorFacade(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def perturb_dataset(
        self,
        dataset: List[DataPoint],
        noise_type: NoiseType,
        noise_proportion: float,
        num_processes: int,
    ) -> List[PerturbedDataPoint]:
        raise NotImplementedError
