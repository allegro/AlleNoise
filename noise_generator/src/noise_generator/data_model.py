from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, NewType, Union


@dataclass
class DataPoint:
    offer_id: str
    text: str
    category_id: str


@dataclass
class PerturbedDataPoint:
    offer_id: str
    text: str
    category_id: str
    original_category_id: str


@dataclass
class Perturbance:
    flip_id: List[str]
    flip_proba: List[float]


class NoiseType(Enum):
    SYMMETRIC = "SYMMETRIC"
    ASYMMETRIC_PAIRFLIP = "ASYMMETRIC_PAIRFLIP"
    ASYMMETRIC_NESTEDFLIP = "ASYMMETRIC_NESTEDFLIP"
    ASYMMETRIC_MATRIXFLIP = "ASYMMETRIC_MATRIXFLIP"


LeafPathMapType = NewType("LeafPathMapType", Dict[str, Dict[str, Union[List[str], List[int]]]])
LeafIdMapType = NewType("LeafIdMapType", Union[Dict, List[Dict]])
CategoryMapType = NewType("CategoryMapType", Dict[str, List[Perturbance]])