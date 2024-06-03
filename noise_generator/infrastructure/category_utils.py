from itertools import chain
from typing import List, Optional

import numpy as np
from anytree import Node
from anytree.search import find_by_attr
from sklearn.metrics import confusion_matrix

from infrastructure.io_utils import load_json
from noise_generator.data_model import LeafIdMapType, LeafPathMapType


def read_category_leaf_path_mapping(
    leaf_path_mapping_file: str
) -> Optional[LeafPathMapType]:
    if leaf_path_mapping_file == "":
        return None

    return load_json(leaf_path_mapping_file)["category_path"]


def create_tree_from_path_list(path_list: List[List[str]]) -> Node:
    root = Node("root")
    for path in path_list:
        c = root
        for elem in path:
            search_result = find_by_attr(c, elem, maxlevel=2)
            if search_result is not None:
                c = search_result
            else:
                c = Node(elem, parent=c)
    return root


def read_category_leaf_id_mapping(leaf_id_mapping_file: str) -> Optional[LeafIdMapType]:
    return load_json(leaf_id_mapping_file) if leaf_id_mapping_file != "" else None


def calculate_confusion_matrix(prediction_file: str) -> Optional[np.ndarray]:
    if prediction_file != "":
        predictions_dump = load_json(prediction_file)
        predictions = list(chain(*[p["indices"] for p in predictions_dump]))
        labels = list(chain(*[p["targets"] for p in predictions_dump]))
        return confusion_matrix(labels, predictions, labels=sorted(set(labels)))
    return None
