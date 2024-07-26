from dataclasses import dataclass


@dataclass
class BaseMetrics:
    accuracy_score: float
    precision_macro: float
    recall_macro: float
    f1_score_macro: float


@dataclass
class LFNDMetrics:
    fraction_clean_correct: float
    fraction_clean_incorrect: float
    fraction_noisy_correct: float
    fraction_noisy_incorrect: float
    fraction_noisy_memorized: float
