import logging
import random
from functools import partial
from multiprocessing import Pool
from typing import List, Tuple

from tqdm import tqdm

from config.noise_generator.noise_generator_defaults import NoiseGeneratorDefaults
from noise_generator.data_model import DataPoint, PerturbedDataPoint


logger = logging.getLogger(__name__)


class SymmetricNoiseGenerator:
    def perturb_dataset(
        self, dataset: List[DataPoint], noise_proportion: float, num_processes: int
    ) -> List[PerturbedDataPoint]:
        dataset_size = len(dataset)

        number_of_points_to_perturb = int(dataset_size * noise_proportion)
        indices_to_affect = random.sample(range(dataset_size), number_of_points_to_perturb)

        categories = self._get_unique_categories(dataset)

        logger.info(
            f"Applying symmetric noise to {noise_proportion * 100}% of {dataset_size} data points in {len(categories)} categories. "
            f"Number of parallel processes: {num_processes}."
        )

        with Pool(num_processes) as pool:
            output = list(
                tqdm(
                    pool.imap(
                        partial(self.perturb_datapoint, indices_to_affect, categories),
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

    @staticmethod
    def perturb_datapoint(
        indices_to_affect: List[int], categories: List[str], pair: Tuple[int, DataPoint]
    ) -> PerturbedDataPoint:
        index, data_point = pair

        if index in indices_to_affect:
            original_category = data_point.category_id
            other_categories = [category for category in categories if category != original_category]
            perturbed_category = other_categories[random.randint(0, len(other_categories) - 1)]

            return PerturbedDataPoint(
                offer_id=data_point.offer_id,
                text=data_point.text,
                category_id=perturbed_category,
                original_category_id=data_point.category_id,
            )

        return PerturbedDataPoint(
            offer_id=data_point.offer_id,
            text=data_point.text,
            category_id=data_point.category_id,
            original_category_id=data_point.category_id,
        )

    def _get_unique_categories(self, dataset: List[DataPoint]) -> List[str]:
        return list(set(map(lambda data_point: data_point.category_id, dataset)))
