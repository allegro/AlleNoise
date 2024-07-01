from __future__ import annotations

import argparse
import logging
import math
import typing as T
from dataclasses import asdict
from typing import Dict, Tuple, Union

import mlflow
import torch
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup

from abstract_classifier.model import AbstractModel, Batch
from bert_classifier.config.defaults import BertTrainingDefaults
from bert_classifier.domain_model.co_teaching import (
    CoTeachingBertClassifierParams,
    CoTeachingPlusUpdateStrategy,
    CoTeachingVariant,
)
from bert_classifier.model.encoder import BertEncoder
from bert_classifier.model.loss import CrossEntropyLoss, Losses, LossReduction
from utils.io_utils import detect_device


logger = logging.getLogger(__name__)


class CoTeachingBertClassifier(AbstractModel):
    hparams: CoTeachingBertClassifierParams

    def __init__(self, *args: T.Any, **kwargs: T.Any) -> None:
        super().__init__(*args, **kwargs)
        self._device = detect_device()
        self.noise_level = self.hparams.prl_spl_coteaching_noise_level
        self.hparams.co_teaching_plus_update_strategy = CoTeachingPlusUpdateStrategy.RECOMMENDED
        self._num_training_steps = -1
        self.automatic_optimization = False

        self._encoder_first_network = BertEncoder(
            encoder_dir=self.hparams.model_path,
            num_labels=self.hparams.num_labels,
            initialize_random_model_weights=self.hparams.initialize_random_model_weights,
        )
        self._encoder_second_network = BertEncoder(
            encoder_dir=self.hparams.model_path,
            num_labels=self.hparams.num_labels,
            initialize_random_model_weights=self.hparams.initialize_random_model_weights,
        )
        self._encoder = self._encoder_first_network

        self.num_steps_per_epoch = kwargs["num_steps_per_epoch"]

        self._loss_no_reduction = self._setup_loss(first_step=True)
        self._loss_mean_reduction = self._setup_loss(first_step=False)

        self._softmax_fn = torch.nn.Softmax(dim=1)
        self.fraction_of_instances_to_keep = self._initialize_fraction_of_instances_to_keep()

    def _setup_loss(self, first_step: bool) -> CrossEntropyLoss:
        if first_step:
            reduction = LossReduction.NONE
        else:
            reduction = LossReduction.MEAN

        return CrossEntropyLoss(ignore_index=-1, reduction=reduction)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._encoder_first_network(token_ids)

    def configure_optimizers(self) -> T.Tuple[Optimizer, LambdaLR]:
        optimizer = AdamW(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = get_linear_schedule_with_warmup(optimizer, self.hparams.warmup_steps, self.num_training_steps)

        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _initialize_fraction_of_instances_to_keep(self) -> float:
        return 1.0 - self.noise_level / float(self.hparams.co_teaching_epoch_k)

    def _is_this_the_last_step_of_epoch(self) -> bool:
        return self.global_step % self.num_steps_per_epoch == self.num_steps_per_epoch - 1

    def _epoch_num_starting_from_one(self) -> int:
        return math.ceil((self.global_step + 1) / self.num_steps_per_epoch)

    def _should_change_instance_keep_rate_function(self, current_epoch_index: int) -> bool:
        return (
            self.hparams.co_teaching_variant == CoTeachingVariant.CO_TEACHING_PLUS
            and self.hparams.co_teaching_plus_update_strategy == CoTeachingPlusUpdateStrategy.RECOMMENDED
            and current_epoch_index > self.hparams.co_teaching_epoch_k
        )

    def _update_fraction_of_instances_to_keep(self) -> None:
        if not self._is_this_the_last_step_of_epoch():
            return

        current_epoch_index_starting_from_one = self._epoch_num_starting_from_one()

        if self._should_change_instance_keep_rate_function(current_epoch_index_starting_from_one):
            first_fraction = current_epoch_index_starting_from_one / self.hparams.co_teaching_epoch_k
            second_fraction = (current_epoch_index_starting_from_one - self.hparams.co_teaching_epoch_k) / (
                self.hparams.num_epochs - self.hparams.co_teaching_epoch_k
            )

            self.fraction_of_instances_to_keep = 1 - min(
                first_fraction * self.noise_level, (1 + second_fraction) * self.noise_level
            )
        else:
            self.fraction_of_instances_to_keep = 1 - min(
                self.noise_level,
                self.noise_level * current_epoch_index_starting_from_one / self.hparams.co_teaching_epoch_k,
            )

        logger.info(f"self.fraction_of_instances_to_keep {self.fraction_of_instances_to_keep}")

    def _select_low_loss_indices(
        self,
        full_loss: torch.Tensor,
        indices_to_select_low_loss_instances: torch.Tensor,
        number_of_kept_instances: int,
        indices: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, Union[torch.Tensor, None]]:
        _, indices_of_low_loss_instances_in_indices_to_select_from = torch.topk(
            full_loss[indices_to_select_low_loss_instances], number_of_kept_instances, axis=0, largest=False
        )
        if indices is not None:
            indices = indices[indices_of_low_loss_instances_in_indices_to_select_from]
        return indices_to_select_low_loss_instances[indices_of_low_loss_instances_in_indices_to_select_from], indices

    def _calculate_loss(self, input, target, indices, requires_indices: bool) -> torch.Tensor:
        if requires_indices:
            return self._loss_mean_reduction(input, target, indices)
        else:
            return self._loss_mean_reduction(input, target)

    def _get_low_loss_candidate_indices(
        self, logits_first_network: torch.Tensor, logits_second_network: torch.Tensor
    ) -> torch.Tensor:
        if self.hparams.co_teaching_variant == CoTeachingVariant.CO_TEACHING_PLUS:
            predictions_first_network = logits_first_network.argmax(axis=1)
            predictions_second_network = logits_second_network.argmax(axis=1)
            disagreement_indices = torch.where(predictions_first_network != predictions_second_network)[0]

            if disagreement_indices.shape[0] == 0:
                logger.info(
                    f"No disagreement between networks, reverting to selecting from all indices (step: {self.global_step})"
                )
                return torch.LongTensor(range(logits_first_network.size(dim=0))).to(self._device)

            return disagreement_indices

        if self.hparams.co_teaching_variant == CoTeachingVariant.CO_TEACHING:
            return torch.LongTensor(range(logits_first_network.size(dim=0))).to(self._device)

        return torch.empty(0)

    def _calculate_first_step_losses(
        self,
        logits_first_network: torch.Tensor,
        logits_second_network: torch.Tensor,
        label: torch.Tensor,
        dataset_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self._loss_no_reduction.requires_indices:
            full_loss_first_network = self._loss_no_reduction(logits_first_network, label, dataset_indices)
            full_loss_second_network = self._loss_no_reduction(logits_second_network, label, dataset_indices)
        else:
            full_loss_first_network = self._loss_no_reduction(logits_first_network, label)
            full_loss_second_network = self._loss_no_reduction(logits_second_network, label)

        return full_loss_first_network, full_loss_second_network

    def _freeze_parameters(self, encoder_name: str) -> None:
        for param_name, param in self.named_parameters():
            if encoder_name in param_name:
                param.requires_grad = False

    def _unfreeze_parameters(self, encoder_name: str) -> None:
        for param_name, param in self.named_parameters():
            if encoder_name in param_name:
                param.requires_grad = True

    def _perform_optimizer_step(
        self, loss_value_first_network: torch.Tensor, loss_value_second_network: torch.Tensor
    ) -> None:
        optimizer = self.optimizers()
        scheduler = self.lr_schedulers()

        optimizer.zero_grad()
        self._freeze_parameters("encoder_second_network")
        self.manual_backward(loss_value_first_network)
        self._unfreeze_parameters("encoder_second_network")
        self._freeze_parameters("encoder_first_network")
        self.manual_backward(loss_value_second_network)
        self._unfreeze_parameters("encoder_first_network")
        optimizer.step()
        scheduler.step()

    def _perform_training_step(
        self, token_ids: torch.Tensor, label: torch.Tensor, dataset_indices: torch.Tensor
    ) -> Dict[Union[int, torch.Tensor]]:
        logits_first_network = self._encoder_first_network(token_ids)
        logits_second_network = self._encoder_second_network(token_ids)

        first_step_loss_first_network, first_step_loss_second_network = self._calculate_first_step_losses(
            logits_first_network, logits_second_network, label, dataset_indices
        )

        low_loss_candidate_indices = self._get_low_loss_candidate_indices(logits_first_network, logits_second_network)

        number_of_observations_to_select_from = low_loss_candidate_indices.shape[0]

        number_of_kept_instances = math.ceil(self.fraction_of_instances_to_keep * number_of_observations_to_select_from)

        (
            low_loss_batch_indices_from_first_network_for_second_network,
            low_loss_dataset_indices_from_first_network_for_second_network,
        ) = self._select_low_loss_indices(
            first_step_loss_first_network, low_loss_candidate_indices, number_of_kept_instances, dataset_indices
        )
        (
            low_loss_batch_indices_from_second_network_for_first_network,
            low_loss_dataset_indices_from_second_network_for_first_network,
        ) = self._select_low_loss_indices(
            first_step_loss_second_network, low_loss_candidate_indices, number_of_kept_instances, dataset_indices
        )

        final_loss_first_network = self._calculate_loss(
            input=logits_first_network[low_loss_batch_indices_from_second_network_for_first_network],
            target=label[low_loss_batch_indices_from_second_network_for_first_network],
            indices=low_loss_dataset_indices_from_second_network_for_first_network,
            requires_indices=self._loss_mean_reduction.requires_indices,
        )

        final_loss_second_network = self._calculate_loss(
            input=logits_second_network[low_loss_batch_indices_from_first_network_for_second_network],
            target=label[low_loss_batch_indices_from_first_network_for_second_network],
            indices=low_loss_dataset_indices_from_first_network_for_second_network,
            requires_indices=self._loss_mean_reduction.requires_indices,
        )

        self._perform_optimizer_step(final_loss_first_network, final_loss_second_network)
        self._update_fraction_of_instances_to_keep()

        probabilities = self._softmax_fn(self._encoder_first_network(token_ids))

        return {
            "epoch": self.current_epoch,
            "probabilities": probabilities,
            "target": label,
        }

    def training_step(self, batch: Batch, batch_index: int) -> T.Dict[str, T.Union[torch.Tensor, int]]:
        token_ids = batch[self.hparams.features_field_name]
        label = batch[self.hparams.label_field_name]
        dataset_indices = batch[self.hparams.index_field_name]

        metrics = self._perform_training_step(token_ids, label, dataset_indices)

        return metrics

    def validation_step(self, batch: Batch, batch_index: int) -> T.Dict[str, torch.Tensor]:
        tokens_ids = batch[self._hparams.features_field_name]
        label = batch[self._hparams.label_field_name]

        logits_first_network = self._encoder_first_network(tokens_ids)
        logits_second_network = self._encoder_second_network(tokens_ids)

        loss_first_network = self._loss_mean_reduction(logits_first_network, label)
        loss_second_network = self._loss_mean_reduction(logits_second_network, label)
        probabilities_first_network = self._softmax_fn(logits_first_network)
        probabilities_second_network = self._softmax_fn(logits_second_network)

        outputs = {
            "loss": loss_first_network,
            "loss_second_network": loss_second_network,
            "probabilities": probabilities_first_network,
            "probabilities_second_network": probabilities_second_network,
            "target": label,
        }

        if self.hparams.lfnd_logging_enabled:
            true_label = batch[self.hparams.true_label_field_name]
            outputs["true_target"] = true_label

        mlflow.log_metric("val_loss_first_network", loss_first_network.item(), self.global_step)
        mlflow.log_metric("val_loss_second_network", loss_second_network.item(), self.global_step)

        return outputs

    @staticmethod
    def from_args(args: argparse.Namespace) -> CoTeachingBertClassifier:
        model_args = asdict(CoTeachingBertClassifierParams())
        model_args.update(**vars(args))

        return CoTeachingBertClassifier(**model_args)

    @staticmethod
    def add_argparse_args(parent_parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--co-teaching", action="store_true", default=BertTrainingDefaults.CO_TEACHING_ENABLED)
        parser.add_argument("--co-teaching-first-step-loss", default=Losses.CROSS_ENTROPY, type=str)
        parser.add_argument("--co-teaching-epoch-k", default=BertTrainingDefaults.CO_TEACHING_EPOCH, type=int)
        parser.add_argument(
            "--co-teaching-plus-update-strategy",
            default=BertTrainingDefaults.CO_TEACHING_PLUS_UPDATE_STRATEGY,
            type=CoTeachingPlusUpdateStrategy,
        )
        parser.add_argument(
            "--co-teaching-variant", default=BertTrainingDefaults.CO_TEACHING_VARIANT, type=CoTeachingVariant
        )

        return parser
