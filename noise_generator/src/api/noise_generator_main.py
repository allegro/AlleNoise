import logging
from argparse import ArgumentParser

from config.noise_generator.noise_generator_defaults import NoiseGeneratorDefaults
from infrastructure.dataset_io import load_dataset, save_perturbed_dataset
from infrastructure.logging_utils import configure_logging
from noise_generator.data_model import NoiseType
from noise_generator.internal.noise_generator_impl import NoiseGeneratorImpl
from noise_generator.noise_generator_facade import NoiseGeneratorFacade


logger = logging.getLogger(__name__)


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", required=True, help="Dataset to perturb", type=str)
    parser.add_argument(
        "--noise-type", required=True, choices=list(NoiseType), help="Noise type to apply", type=NoiseType
    )
    parser.add_argument("--noise-proportion", required=True, help="Noise proportion, ranged (0.0, 1.0]", type=float)
    parser.add_argument("--output-path", required=True, help="Output file path", type=str)
    parser.add_argument("--seed", required=False, default=NoiseGeneratorDefaults.SEED, help="Rng seed", type=int)
    parser.add_argument(
        "--num-processes",
        help="Number of processes for parallel noise generation",
        default=NoiseGeneratorDefaults.NUMBER_OF_PROCESSES,
        type=int,
    )
    parser.add_argument(
        "--category-id-mapping-file",
        required=False,
        default="",
        help="Mapping between category leaf_ids and token ids (json)",
        type=str,
    )
    parser.add_argument(
        "--category-path-mapping-file",
        required=False,
        default="",
        help="Mapping between category leaf_ids and category paths (json)",
        type=str,
    )
    parser.add_argument(
        "--model-predictions-file", required=False, default="", help="Model predictions on test data (json)", type=str
    )
    parser.add_argument(
        "--asymmetric-category-level",
        required=False,
        default=NoiseGeneratorDefaults.PERTURBANCE_CATEGORY_LEVEL,
        help="Category tree level on which label corruption is applied",
        type=int,
    )
    return parser.parse_args()


def make_noise_generator(arguments) -> NoiseGeneratorFacade:
    return NoiseGeneratorImpl(arguments)


if __name__ == "__main__":
    configure_logging()
    arguments = parse_arguments()
    dataset = load_dataset(arguments.dataset_path)
    noise_generator = make_noise_generator(arguments)
    perturbed_dataset = noise_generator.perturb_dataset(
        dataset, arguments.noise_type, arguments.noise_proportion, arguments.num_processes
    )
    save_perturbed_dataset(perturbed_dataset, arguments.output_path)
