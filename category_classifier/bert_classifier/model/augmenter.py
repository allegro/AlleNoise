import copy
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import List, Tuple, TypeVar

import numpy as np
import torch

from bert_classifier.config.constants import TokenizerConstants


AugmenterType = TypeVar("C", bound="Transformation")


class AugmentationType(Enum):
    DROP_RANDOM = 1
    DROP_CONSECUTIVE = 2
    SWAP = 3


@dataclass
class AugmentationDefinition:
    type: AugmentationType
    fraction: float


def _augmentation_definition_parser(definition: str) -> List[AugmentationDefinition]:
    split_definitions = definition.split(",")
    split_parameters = (single_definition.split("-") for single_definition in split_definitions)
    transformations = [
        AugmentationDefinition(type=AugmentationType[name], fraction=float(parameter))
        for name, parameter in split_parameters
    ]
    return transformations


class Transformation(ABC):
    def __init__(self, output_length):
        self._output_length = output_length

    @abstractmethod
    def transform(self, tokens: torch.Tensor) -> torch.Tensor:
        ...

    @staticmethod
    def add_padding(
        tokens: List[int],
        output_length: int,
        pad_token_id: int = TokenizerConstants.PAD_TOKEN_ID,
    ) -> torch.Tensor:
        token_len = len(tokens)
        return torch.nn.functional.pad(
            torch.tensor(tokens, dtype=torch.int64),
            (0, output_length - token_len),
            value=pad_token_id,
        )

    @staticmethod
    def remove_padding(tokens: torch.Tensor, pad_token_id: int = TokenizerConstants.PAD_TOKEN_ID) -> List[int]:
        token_list = tokens.tolist()
        if pad_token_id in token_list:
            first_pad_idx = token_list.index(pad_token_id)
            token_list = token_list[:first_pad_idx]
        return token_list

    def preprocess(self, tokens: torch.Tensor) -> List[int]:
        return self.remove_padding(tokens)

    def postprocess(self, tokens: List[int]) -> torch.Tensor:
        token_len = len(tokens)
        if token_len < self._output_length:
            return self.add_padding(tokens, self._output_length)
        return torch.tensor(tokens, dtype=torch.int64)


class DropToken(Transformation):
    """
    Augmentation through token dropping.
        * consecutive dropping - subsequence of N consecutive tokens is dropped from the input sequence
        * random dropping - N arbitrary tokens are dropped from the input sequence
    param:
        - drop_fraction - ratio of tokens to drop with respect to the length of the sequence [0-1]
        - drop_type - ['consecutive', 'random']
        - output_length - length of padded token list
    """

    name = "drop"

    def __init__(self, definition: AugmentationDefinition, output_length: int):
        self.drop_fraction = definition.fraction
        self.drop_type = definition.type
        self.indices_sampling_mapping = {
            AugmentationType.DROP_CONSECUTIVE: self.sample_consecutive_indices,
            AugmentationType.DROP_RANDOM: self.sample_random_indices,
        }
        super().__init__(output_length)

    @staticmethod
    def sample_consecutive_indices(token_len: int, num_samples_to_drop: int) -> List[int]:
        possible_first_index = range(token_len - num_samples_to_drop + 1)
        first_index_to_drop = np.random.choice(possible_first_index)
        indices_to_drop = range(first_index_to_drop, first_index_to_drop + num_samples_to_drop)
        return list(indices_to_drop)

    @staticmethod
    def sample_random_indices(token_len: int, num_samples_to_drop: int) -> List[int]:
        indices_to_drop = np.random.choice(range(token_len), num_samples_to_drop, replace=False)
        return list(indices_to_drop)

    def transform(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.preprocess(tokens)
        token_len = len(tokens)
        num_samples_to_drop = int(token_len * self.drop_fraction)
        indices_to_drop = self.indices_sampling_mapping[self.drop_type](token_len, num_samples_to_drop)
        transformed_tokens = [token for idx, token in enumerate(tokens) if idx not in indices_to_drop]
        return self.postprocess(transformed_tokens)


class SwapToken(Transformation):
    """
    Augmentation through token swapping.
    param:
        - swap_fraction - ratio of tokens that change positions by pair swaps [0, 1]
        - output_length - length of padded token list
    """

    name = "swap"

    def __init__(self, definition: AugmentationDefinition, output_length: int):
        self.swap_fraction = definition.fraction
        super().__init__(output_length)

    @staticmethod
    def _swap_list_elements(input_list: List[int], pair_list: List[Tuple[int, int]]) -> List[int]:
        input, pairs = np.array(input_list), np.array(pair_list)
        swapped = copy.deepcopy(input)
        if pairs.shape[0] != 0:
            swapped[pairs[:, 0]] = input[pairs[:, 1]]
            swapped[pairs[:, 1]] = input[pairs[:, 0]]
        return swapped.tolist()

    @staticmethod
    def sample_swap_indices(token_len: int, num_pairs_to_swap: int) -> List[Tuple[int, int]]:
        swaps = np.random.choice(range(token_len), (num_pairs_to_swap, 2), replace=False)
        return list(map(tuple, swaps))

    def swap(self, tokens: List[int], num_pairs_to_swap: int) -> List[int]:
        pairs_to_swap = self.sample_swap_indices(len(tokens), num_pairs_to_swap)
        transformed_tokens = self._swap_list_elements(tokens, pairs_to_swap)
        return transformed_tokens

    def transform(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = self.preprocess(tokens)
        num_pairs_to_swap = int(len(tokens) * self.swap_fraction / 2)
        transformed_tokens = self.swap(tokens, num_pairs_to_swap)
        return self.postprocess(transformed_tokens)


class Augmenter:
    def __init__(self, parsed_definitions: List[AugmentationDefinition], output_length: int):
        self.output_length = output_length
        self.num_augmentations = len(parsed_definitions)
        self.augmentation_mapping = {
            AugmentationType.DROP_RANDOM: DropToken,
            AugmentationType.DROP_CONSECUTIVE: DropToken,
            AugmentationType.SWAP: SwapToken,
        }
        self.transformations = self._setup_transformations(parsed_definitions)

    def _setup_transformations(self, parsed_definitions: List[AugmentationDefinition]) -> List[AugmenterType]:
        return [
            self.augmentation_mapping[definition.type](definition, self.output_length)
            for definition in parsed_definitions
        ]

    def augment(self, tokens: torch.Tensor) -> torch.Tensor:
        transformed_tokens = torch.zeros(self.num_augmentations, self.output_length, dtype=torch.int64)
        for idx, transformation in enumerate(self.transformations):
            transformed_tokens[idx, :] = transformation.transform(tokens)
        return transformed_tokens
