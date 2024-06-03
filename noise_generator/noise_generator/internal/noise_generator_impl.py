import argparse
from typing import List

from noise_generator.data_model import DataPoint, NoiseType, PerturbedDataPoint
from noise_generator.internal.symmetric.symmetric_noise_generator import SymmetricNoiseGenerator
from noise_generator.noise_generator.class_dependent_noise import ClassDependentNoiseGenerator
from noise_generator.noise_generator_facade import NoiseGeneratorFacade


class NoiseGeneratorImpl(NoiseGeneratorFacade):
    def __init__(self, arguments: argparse.Namespace):
        super().__init__()
        self.noise_generators = {
            NoiseType: ClassDependentNoiseGenerator(arguments),
        }

    def perturb_dataset(
        self,
        dataset: List[DataPoint],
        noise_type: NoiseType,
        noise_proportion: float,
        num_processes: int,
    ) -> List[PerturbedDataPoint]:
        if noise_type == NoiseType.SYMMETRIC:
            symmetric_noise_generator = SymmetricNoiseGenerator()

            return symmetric_noise_generator.perturb_dataset(
                dataset=dataset, noise_proportion=noise_proportion, num_processes=num_processes
            )
        elif noise_type in NoiseType:
            return self.noise_generators[type(noise_type)].perturb_dataset(
                dataset=dataset,
                noise_type=noise_type,
                noise_proportion=noise_proportion,
                num_processes=num_processes,
            )


        return list(
            map(
                lambda data_point: PerturbedDataPoint(
                    offer_id=data_point.offer_id,
                    text=data_point.text,
                    category_id=data_point.category_id,
                    original_category_id=data_point.category_id,
                ),
                dataset,
            )
        )
