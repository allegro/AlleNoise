import enum
import math
from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn.functional as F

from pcs_category_classifier.bert_classifier.model.loss_helpers import (
    DropInstancesWithTopValues,
    LossUsingCrossEntropyGradientNorm,
)
from pcs_category_classifier.utils.io_utils import detect_device


class Losses:
    CROSS_ENTROPY = "cross-entropy"
    PRL_L_CROSS_ENTROPY = "prl-l-cross-entropy"
    SPL_CROSS_ENTROPY = "spl-cross-entropy"
    CLIPPED_CROSS_ENTROPY = "clipped-cross-entropy"
    ELR_CROSS_ENTROPY = "elr-cross-entropy"
    GJSD_LOSS = "gjsd-loss"


class LossReduction(enum.Enum):
    MEAN = "mean"
    NONE = "none"


class LossFunction(ABC):
    @abstractmethod
    def requires_indices(self) -> bool:
        ...

    @staticmethod
    def get_reduced_values(values: torch.Tensor, reduction: LossReduction = LossReduction.NONE) -> torch.Tensor:
        if reduction == LossReduction.MEAN:
            return values.mean()
        return values


class CrossEntropyLoss(LossFunction, torch.nn.CrossEntropyLoss):
    def __init__(self, ignore_index: int = -1, reduction: LossReduction = LossReduction.MEAN):
        torch.nn.CrossEntropyLoss.__init__(self, ignore_index=ignore_index, reduction=reduction.value)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return super().forward(input, target)

    @property
    def requires_indices(self) -> bool:
        return False


class ClippedCrossEntropyLoss(LossFunction, torch.nn.CrossEntropyLoss):
    """
    Implements the idea from the seminar: what would happen if we only clipped loss,
    and not removed observations with large loss/norm of gradient of loss:
    clipping where norm of the loss value exceeds the threshold
    """

    def __init__(
        self,
        ignore_index: int = -1,
        clip_at_value: float = 0,
        reduction: LossReduction = LossReduction.MEAN,
        start_from_epoch: int = 0,
    ):
        torch.nn.CrossEntropyLoss.__init__(self, ignore_index=ignore_index, reduction="none")
        self.clip_at_value = clip_at_value
        self.loss_reduction = reduction
        self.start_from_epoch = start_from_epoch

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = super().forward(input, target)
        clipped_loss = torch.where(loss < self.clip_at_value, loss, torch.ones_like(loss) * self.clip_at_value)
        return self.get_reduced_values(clipped_loss, self.loss_reduction)

    @property
    def requires_indices(self) -> bool:
        return False


class PrlLCrossEntropyLoss(
    LossFunction, torch.nn.CrossEntropyLoss, DropInstancesWithTopValues, LossUsingCrossEntropyGradientNorm
):
    """
    Implements PRL(L) method from http://proceedings.mlr.press/v139/liu21v/liu21v.pdf
    """

    def __init__(self, noise_level: float, ignore_index: int = -1):
        torch.nn.CrossEntropyLoss.__init__(self, ignore_index=ignore_index, reduction="none")
        DropInstancesWithTopValues.__init__(self, dropped_fraction=noise_level, loss_func=super().forward)
        LossUsingCrossEntropyGradientNorm.__init__(self)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_gradient_norm = self._cross_entropy_loss_gradient_norm(input, target)
        return self.mean_loss_of_kept_instances(input, target, loss_gradient_norm)

    @property
    def requires_indices(self) -> bool:
        return False


class SplCrossEntropyLoss(LossFunction, torch.nn.CrossEntropyLoss, DropInstancesWithTopValues):
    """
    Implements Self-paced learning variant for learning from noisy data.
    Based on implementation from Learning Deep Neural Networks under Agnostic Corrupted Supervision
    https://github.com/illidanlab/PRL/blob/main/loss.py
    and description from the paper:
    SPL (Jiang et al.,2018): self-paced learning (also known as the trimmed loss
    or predefined curriculum) by dropping the data points with large losses ...
    """

    def __init__(self, noise_level: float, ignore_index: int = -1):
        torch.nn.CrossEntropyLoss.__init__(self, ignore_index=ignore_index, reduction="none")
        DropInstancesWithTopValues.__init__(self, dropped_fraction=noise_level, loss_func=super().forward)

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = super().forward(input, target)
        return self.mean_loss_of_kept_instances(input, target, loss)

    @property
    def requires_indices(self) -> bool:
        return False


class ElrCrossEntropyLoss(LossFunction, torch.nn.CrossEntropyLoss):
    """
    Implements Early Learning Regularization from
    Early-Learning Regularization Prevents Memorization of Noisy Labels, NeurIPS 2020
    Loss is implemented based on Equation (6), and targets are estimated as in Section 4.3, Equation (9), and on the
    official implementation https://github.com/shengliu66/ELR/blob/master/ELR/model/loss.py
    """

    def __init__(
        self,
        num_observations: int,
        num_classes: int,
        targets_momentum_beta: float,
        regularization_constant_lambda: float,
        clamp_margin: float,
        reduction: LossReduction = LossReduction.NONE,
        ignore_index: int = -1,
    ):
        torch.nn.CrossEntropyLoss.__init__(self, ignore_index=ignore_index, reduction=reduction.value)
        self.num_classes = num_classes
        self._device = detect_device()
        self.target = torch.zeros(num_observations, num_classes).to(self._device)
        self.targets_momentum_beta = targets_momentum_beta
        self.regularization_constant_lambda = regularization_constant_lambda
        self.clamp_margin = clamp_margin
        self._softmax = torch.nn.Softmax(dim=1)

    def forward(
        self, input: torch.Tensor, target: torch.Tensor, indices: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cross_entropy_loss = super().forward(input, target)
        cross_entropy_term = cross_entropy_loss.mean()
        if indices is None:
            return cross_entropy_term
        indices = indices.long().to(self._device)
        probabilities = self._softmax(input)

        # Clamping is not mentioned in the paper, only given in the implementation
        clamped_probabilities = torch.clamp(probabilities, self.clamp_margin, 1.0 - self.clamp_margin)
        detached_clamped_probabilities = clamped_probabilities.data.detach()
        detached_normalized_probabilities = detached_clamped_probabilities / detached_clamped_probabilities.sum(
            dim=1, keepdim=True
        )

        self.target[indices] = self.targets_momentum_beta * self.target[indices] + (
            1 - self.targets_momentum_beta
        ) * detached_normalized_probabilities.to(self.target.device)

        # Based on https://github.com/shengliu66/ELR/blob/c93fe057c1a3d898355a25763eb470eb31bab9ef/ELR/model/loss.py#L27
        elr_regularization = (
            (1 - (self.target[indices].to(input.device) * clamped_probabilities).sum(dim=1)).log()
        ).mean()
        final_loss = cross_entropy_term + self.regularization_constant_lambda * elr_regularization
        return final_loss, cross_entropy_term.detach(), elr_regularization.detach()

    @property
    def requires_indices(self) -> bool:
        return True


class JensenShannonDivergenceLoss(LossFunction, torch.nn.Module):
    """
    Implements Generalized Jensen Shannon Loss from
    "Generalized Jensen-Shannon Divergence Loss for Learning with Noisy Labels" NeurIPS 2021
    official implementation: https://github.com/ErikEnglesson/GJS
        * own implementation of Kullback-Leibler divergence
        * facilitate loss calculation for 3 scenarios:
            * M=2 consistency_regularization=False
            * M>2 consistency_regularization=True/False
            disabled consistency regularization enforce averaging of predictions
    param:
        - num_distributions - number of independent distributions equal to num_augmentations+1 [2, 3...]
        - pi_weight - scalar parameter used in weighting average of the distributions [0, 1]
        - consistency_regularization - controls the input distributions for the loss, when enabled GJSD receive
        single weighted average of input distributions [bool]
    Additional remarks: Throughout experiments, we were unable to reproduce one of the properties of GJSD,
    the loss being the generalization of MAE.
    """

    def __init__(
        self,
        num_classes: int,
        num_distributions: int,
        pi_weight: float,
        consistency_regularization: bool,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.M_num_distributions = num_distributions if consistency_regularization else 2
        self.pi_weights, self.pi_weights_average, self.Z_scaling_factor = self.calculate_weights(
            pi_weight, num_distributions
        )
        self.consistency_regularization = consistency_regularization and (num_distributions > 2)

    @staticmethod
    def calculate_weights(first_pi_weight: float, num_distributions: int) -> Tuple[torch.Tensor, torch.Tensor, float]:
        pi_weights = [first_pi_weight]
        pi_weights.extend([(1.0 - first_pi_weight) / (num_distributions - 1) for _ in range(num_distributions - 1)])
        pi_weights_average = [first_pi_weight, 1.0 - first_pi_weight]
        scaling_factor = -(1.0 - first_pi_weight) * math.log(1.0 - first_pi_weight)
        return torch.Tensor(pi_weights), torch.Tensor(pi_weights_average), scaling_factor

    @staticmethod
    def calculate_kullback_leibler_divergence(
        prediction: torch.Tensor, target: torch.Tensor, clamp_margin: float = 1e-7
    ) -> torch.Tensor:
        """
        Implements D_KL(X || Y) = x *          ( log(x)          - log(y) )      = D_KL(X || Y)
                                = prediction * ( log(prediction) - log(target) ) = D_KL(prediction || target) =
                                = p *          ( log(p)          - log(m) )      = D_KL(p || m)
        """
        log_target = target.clamp(min=clamp_margin, max=1.0).log()
        output = prediction * (prediction.clamp(min=clamp_margin).log() - log_target)
        output_without_negative_values = torch.where(prediction > 0, output, torch.zeros_like(output))
        return torch.sum(output_without_negative_values, axis=1).mean()

    def calculate_jensen_shannon_divergence(self, p_distributions: torch.Tensor) -> torch.Tensor:
        if self.consistency_regularization:
            pi_weights = self.pi_weights.to(p_distributions.device)
        else:
            pi_weights = self.pi_weights_average.to(p_distributions.device)

        weighted_mean_distribution = torch.sum(torch.mul(pi_weights, p_distributions), dim=2)
        dkl_scores = [
            self.calculate_kullback_leibler_divergence(p_distributions[:, :, i], weighted_mean_distribution)
            for i in range(self.M_num_distributions)
        ]
        weighted_scores = torch.mul(pi_weights, torch.stack(dkl_scores))
        return weighted_scores.sum()

    def compute_average_predictions(self, predictions: torch.Tensor) -> torch.Tensor:
        pi_weights = self.pi_weights.to(predictions.device)
        avg_predictions = torch.mul(pi_weights[1:], predictions).sum(axis=2).unsqueeze(dim=2)
        return avg_predictions

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if len(target.shape) == 1:
            # mixup already performs one_hot encoding
            target = F.one_hot(target, num_classes=self.num_classes).float()
        target = target.unsqueeze(dim=2)

        if len(input.shape) == 2:
            # assure shape for non-augmented sequences
            input = input.unsqueeze(dim=2)
        predictions = F.softmax(input, dim=1)
        if not self.consistency_regularization:
            predictions = self.compute_average_predictions(predictions)

        p_distributions = torch.cat([target, predictions], dim=2)
        loss = self.calculate_jensen_shannon_divergence(p_distributions) / self.Z_scaling_factor
        return loss

    @property
    def requires_indices(self) -> bool:
        return False
