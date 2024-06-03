from typing import List

import numpy as np
from anytree.search import find_by_attr

from infrastructure.category_utils import create_tree_from_path_list
from noise_generator.data_model import (
    CategoryMapType,
    NoiseType,
    LeafIdMapType,
    LeafPathMapType,
    Perturbance,
)


class LabelMappingGenerator:
    def __init__(self):
        self.noise_type_mapping = {
            NoiseType.SYMMETRIC: self.compute_symmetric_mapping,
            NoiseType.ASYMMETRIC_PAIRFLIP: self.compute_asymmetric_pairflip_mapping,
            NoiseType.ASYMMETRIC_MATRIXFLIP: self.compute_asymmetric_matrixflip_mapping,
            NoiseType.ASYMMETRIC_NESTEDFLIP: self.compute_asymmetric_nestedflip_mapping,
        }

    def _uniform_proba(self, n: int) -> List[float]:
        return self._weighted_proba(np.ones(n))

    @staticmethod
    def _weighted_proba(occurrence_rate: np.ndarray) -> List[float]:
        return (occurrence_rate / np.sum(occurrence_rate)).tolist()

    @staticmethod
    def create_mapping_from_perturbance_definitions(
        unique_categories: List[str], perturbance_definition: List[Perturbance]
    ) -> CategoryMapType:
        return dict(zip(unique_categories, perturbance_definition))

    def compute_symmetric_mapping(self, unique_categories: List[str]) -> CategoryMapType:
        complementary_categories = [list(set(unique_categories) - set([x])) for x in unique_categories]
        proba = self._uniform_proba(len(unique_categories) - 1)
        perturbance_definition = [Perturbance(c, proba) for c in complementary_categories]
        return self.create_mapping_from_perturbance_definitions(unique_categories, perturbance_definition)

    def compute_asymmetric_pairflip_mapping(self, unique_categories: List[str]) -> CategoryMapType:
        shifted_categories = np.roll(unique_categories, shift=-1).tolist()
        perturbance_definition = [Perturbance([c], [1]) for c in shifted_categories]
        return self.create_mapping_from_perturbance_definitions(unique_categories, perturbance_definition)

    def compute_asymmetric_matrixflip_mapping(
        self, unique_categories: List[str], confusion_matrix: np.ndarray, leaf_id_mapping: LeafIdMapType
    ) -> CategoryMapType:
        error_mask = (confusion_matrix >= 1) * (np.eye(confusion_matrix.shape[0]) != 1)
        error_indices = np.where(error_mask)
        id2category = {v: k for k, v in leaf_id_mapping.items()}
        flippable_categories = [id2category[c] for c in np.unique(error_indices[0])]
        perturbance_definition = []
        for category in unique_categories:
            if category in flippable_categories:
                category_id = leaf_id_mapping.get(category)
                category_indices = np.where(error_indices[0] == category_id)
                error_predictions = error_indices[1][category_indices]
                perturbed_leaf_ids = [id2category[c] for c in error_predictions]
                error_rates = confusion_matrix[category_id, error_predictions]
                proba = self._weighted_proba(error_rates)
            else:
                perturbed_leaf_ids = None
                proba = None
            perturbance_definition.append(Perturbance(perturbed_leaf_ids, proba))
        return self.create_mapping_from_perturbance_definitions(unique_categories, perturbance_definition)

    def compute_asymmetric_nestedflip_mapping(
        self, unique_categories: List[str], leaf_path_mapping: LeafPathMapType, category_perturbance_level: int
    ) -> CategoryMapType:
        category_tree = create_tree_from_path_list(list(leaf_path_mapping.values()))
        perturbance_definition = []
        for category in unique_categories:
            if len(leaf_path_mapping.get(category)) > category_perturbance_level:
                node = find_by_attr(category_tree, category)
                on_the_same_branch = list(node.parent.children)
                if len(on_the_same_branch) > 1:
                    leaf_index = on_the_same_branch.index(node)
                    next_index = leaf_index + 1 if leaf_index < len(on_the_same_branch) - 1 else 0
                    perturbed_leaf_ids = [l.name for l in on_the_same_branch[next_index].leaves]
                    proba = self._uniform_proba(len(perturbed_leaf_ids))
                else:
                    perturbed_leaf_ids = None
                    proba = None
            else:
                perturbed_leaf_ids = None
                proba = None
            perturbance_definition.append(Perturbance(perturbed_leaf_ids, proba))
        return self.create_mapping_from_perturbance_definitions(unique_categories, perturbance_definition)
