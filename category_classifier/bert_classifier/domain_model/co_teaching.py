import typing as T
from dataclasses import dataclass
from enum import Enum


class CoTeachingVariant(Enum):
    CO_TEACHING_PLUS = "CO_TEACHING_PLUS"
    CO_TEACHING = "CO_TEACHING"


class CoTeachingPlusUpdateStrategy(Enum):
    SIMPLE = "SIMPLE"
    RECOMMENDED = "RECOMMENDED"


@dataclass
class CoTeachingBertClassifierParams:
    num_labels: int = 0
    features_field_name: str = ""
    label_field_name: str = ""
    use_mc_dropout: T.Optional[bool] = None
    mc_dropout_trials: T.Optional[int] = None
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    weight_decay: float = 0.0
    loss: str = "cross-entropy"
    prl_spl_coteaching_noise_level: float = 0.0
    model_path: str = None
    load_pretrained_weights: bool = False
    co_teaching_variant: CoTeachingVariant = CoTeachingVariant.CO_TEACHING
    co_teaching_plus_update_strategy: CoTeachingPlusUpdateStrategy = CoTeachingPlusUpdateStrategy.RECOMMENDED
