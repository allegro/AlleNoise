import argparse
import logging
import random
from functools import partial
from multiprocessing import Pool
from typing import Dict, List, Tuple, Union

import numpy as np
from tqdm import tqdm

from config.noise_generator.noise_generator_defaults import NoiseGeneratorDefaults
from infrastructure.category_utils import (
    calculate_confusion_matrix,
    read_category_leaf_id_mapping,
    read_category_leaf_path_mapping,
)
from noise_generator.abstract_noise_generator import AbstractNoiseGenerator
from noise_generator.data_model import NoiseType, DataPoint, LeafPathMapType, PerturbedDataPoint
from noise_generator.label_transition_transforms import LabelMappingGenerator


logger = logging.getLogger(__name__)


class ClassDependentNoiseGenerator(AbstractNoiseGenerator):
    def __init__(self, arguments: argparse.Namespace):
        super().__init__()
        self.hparams = arguments
        self.confusion_matrix, self.leaf_id_mapping, self.leaf_path_mapping = self._setup_artifacts()

    def _setup_artifacts(self) -> Tuple[np.ndarray, Union[Dict, List[Dict]], LeafPathMapType]:
        confusion_matrix = calculate_confusion_matrix(self.hparams.model_predictions_file)
        leaf_id_mapping = read_category_leaf_id_mapping(self.hparams.category_id_mapping_file)
        leaf_path_mapping = read_category_leaf_path_mapping(
            leaf_path_mapping_file=self.hparams.category_path_mapping_file
        )
        return confusion_matrix, leaf_id_mapping, leaf_path_mapping

    def perturb_dataset(
        self,
        dataset: List[DataPoint],
        noise_type: NoiseType,
        noise_proportion: float,
        num_processes: int,
    ) -> List[PerturbedDataPoint]:
        dataset_size = len(dataset)
        categories = self._get_unique_categories(dataset)
        category_mapping = self._create_category_mapping(noise_type, categories)
        indices_to_affect = self._choose_indices_to_perturb(dataset, noise_type, noise_proportion, category_mapping)

        logger.info(
            f"Applying {noise_type.value} noise to {noise_proportion * 100}% of {dataset_size} data points in"
            f" {len(categories)} categories. "
            f"Number of parallel processes: {num_processes}."
        )

        with Pool(num_processes) as pool:
            output = list(
                tqdm(
                    pool.imap(
                        partial(self.perturb_datapoint, indices_to_affect, category_mapping, seed=self.hparams.seed),
                        enumerate(dataset),
                        chunksize=int(
                            dataset_size / num_processes / NoiseGeneratorDefaults.NUMBER_OF_CHUNKS_PER_PROCESS
                        ),
                    ),
                    total=dataset_size,
                )
            )

        logger.info("Done.")
        return output

    def perturb_datapoint(
        self, indices_to_affect: List[int], category_mapping: LeafPathMapType, pair: Tuple[int, DataPoint], seed: int
    ) -> PerturbedDataPoint:
        index, data_point = pair
        if index in indices_to_affect:
            random.seed(seed)
            perturbed_category = random.choices(
                population=category_mapping.get(data_point.category_id).flip_id,
                weights=category_mapping.get(data_point.category_id).flip_proba,
                k=1,
            )[0]
            return self.label_perturbance(data_point, perturbed_category)

        return self.identity_perturbance(data_point)

    def _random_sample_from_allowed_categories(
        self,
        dataset: List[DataPoint],
        dataset_size: int,
        category_mapping: LeafPathMapType,
        number_of_points_to_perturb: int,
    ):
        forbidden_categories = [k for k in category_mapping.keys() if category_mapping.get(k).flip_id == None]
        dataset_categories = self._get_dataset_categories(dataset)
        possible_sampling_indices = [
            i for i in range(dataset_size) if dataset_categories[i] not in forbidden_categories
        ]
        if len(possible_sampling_indices) >= number_of_points_to_perturb:
            indices_to_affect = random.sample(possible_sampling_indices, number_of_points_to_perturb)
        else:
            raise RuntimeError(
                f"Sampling size ({number_of_points_to_perturb}) out of range after excluding "
                f"forbidden samples ({len(possible_sampling_indices)})"
            )
        return indices_to_affect

    def _choose_indices_to_perturb(
        self,
        dataset: List[DataPoint],
        noise_type: NoiseType,
        noise_proportion: float,
        category_mapping: LeafPathMapType,
    ) -> List[int]:
        dataset_size = len(dataset)
        number_of_points_to_perturb = int(dataset_size * noise_proportion)
        if noise_type in [NoiseType.SYMMETRIC, NoiseType.ASYMMETRIC_PAIRFLIP]:
            indices_to_affect = random.sample(range(dataset_size), number_of_points_to_perturb)
        elif noise_type in [
            NoiseType.ASYMMETRIC_NESTEDFLIP,
            NoiseType.ASYMMETRIC_MATRIXFLIP,
        ]:
            indices_to_affect = self._random_sample_from_allowed_categories(
                dataset, dataset_size, category_mapping, number_of_points_to_perturb
            )
        else:
            raise (NotImplementedError)
        return indices_to_affect

    def _create_category_mapping(
        self,
        noise_type: NoiseType,
        unique_categories: List[str],
    ) -> LeafPathMapType:
        category_mapper = LabelMappingGenerator()
        if noise_type in [NoiseType.SYMMETRIC, NoiseType.ASYMMETRIC_PAIRFLIP]:
            category_mapping = category_mapper.noise_type_mapping[noise_type](unique_categories)

        elif noise_type == NoiseType.ASYMMETRIC_NESTEDFLIP:
            category_mapping = category_mapper.noise_type_mapping[noise_type](
                unique_categories, self.leaf_path_mapping, self.hparams.asymmetric_category_level
            )

        elif noise_type == NoiseType.ASYMMETRIC_MATRIXFLIP:
            category_mapping = category_mapper.noise_type_mapping[noise_type](
                unique_categories, self.confusion_matrix, self.leaf_id_mapping
            )

        else:
            raise (NotImplementedError)

        return category_mapping
