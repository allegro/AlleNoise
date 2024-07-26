import typing as T

import numpy as np
import torch


def add_mixup_samples(
    feature: torch.Tensor, label: torch.Tensor, alpha: float = 0.2, ratio: float = 0.2
) -> T.Tuple[torch.Tensor, torch.Tensor]:
    """
    Implements MixUp technique from
    "mixup: Beyond Empirical Risk Minimization", ICLR 2018
        * in-batch augmentation
        * lambda fixed per batch
        * mixed pairs sampled without replacement
    param:
        - alpha: defines Beta distribution beta(alpha, alpha)
                 controlling mixing magnitude lambda [0-1]
        - ratio: augmentation size defined as % of batch size [0-0.5]
    """
    batch_size = label.shape[0]
    mixup_size = int(batch_size * ratio)
    mixup_idx = list(range(batch_size))
    mixup_pairs = np.random.choice(mixup_idx, size=(2, mixup_size), replace=False)

    augment_features, mixing_features = feature[mixup_pairs, :]
    augment_labels, mixing_labels = label[mixup_pairs, :]
    mixup_features, mixup_labels = _mixup_samples(
        (augment_features, augment_labels), (mixing_features, mixing_labels), alpha=alpha
    )

    return mixup_features, mixup_labels


def _mixup_samples(
    samples: T.Tuple[torch.Tensor, torch.Tensor], mixup_samples: T.Tuple[torch.Tensor, torch.Tensor], alpha: float = 0.2
) -> T.Tuple[torch.Tensor, torch.Tensor]:
    lambda_scalar = np.random.beta(alpha, alpha)  # Fixed per batch
    features = lambda_scalar * samples[0] + (1.0 - lambda_scalar) * mixup_samples[0]
    labels = lambda_scalar * samples[1] + (1.0 - lambda_scalar) * mixup_samples[1]
    return features, labels
