from __future__ import annotations

import argparse
import logging
import typing as T
from dataclasses import asdict, dataclass

import torch
import torch.nn.functional as F
from torch.optim import Optimizer
from transformers import AdamW, get_linear_schedule_with_warmup

from category_classifier.abstract_classifier.model import AbstractModel, Batch
from category_classifier.bert_classifier.config.constants import BertTrainingConstants, TokenizerConstants
from category_classifier.bert_classifier.config.defaults import BertTrainingDefaults
from category_classifier.bert_classifier.model.augmenter import Augmenter, _augmentation_definition_parser
from category_classifier.bert_classifier.model.encoder import BertEncoder
from category_classifier.bert_classifier.model.loss import (
    ClippedCrossEntropyLoss,
    CrossEntropyLoss,
    ElrCrossEntropyLoss,
    JensenShannonDivergenceLoss,
    Losses,
    LossFunction,
    PrlLCrossEntropyLoss,
    SplCrossEntropyLoss,
)
from category_classifier.bert_classifier.model.loss_helpers import DropInstancesWithTopValues
from category_classifier.bert_classifier.model.mixup import add_mixup_samples
from category_classifier.bert_classifier.project_setting import ProjectSettings
from category_classifier.utils.io_utils import detect_device
from category_classifier.utils.py_lighting_utils import get_grouped_parameters_with_weight_decay


logger = logging.getLogger(__name__)


@dataclass
class BertClassifierParams:
    num_labels: int = 0
    features_field_name: str = ""
    label_field_name: str = ""
    learning_rate: float = 2e-5
    warmup_steps: int = 100
    weight_decay: float = 0.0
    loss: str = "cross-entropy"
    noise_level: float = 0.0
    model_path: str = None


class BertClassifier(AbstractModel):
    hparams: BertClassifierParams

    def __init__(self, *args: T.Any, **kwargs: T.Any) -> None:
        super().__init__(*args, **kwargs)
        self._device = detect_device()
        self._num_training_steps = -1
        self._encoder = BertEncoder(
            encoder_dir=self.hparams.model_path,
            num_labels=self.hparams.num_labels,
            initialize_random_model_weights=self.hparams.initialize_random_model_weights,
        )

        self._loss_fn = self._setup_loss()
        self._softmax_fn = torch.nn.Softmax(dim=1)
        self._augmenter = self._setup_augmenter()

    def _setup_loss(self) -> LossFunction:
        if self.hparams.loss == Losses.PRL_L_CROSS_ENTROPY:
            logger.info("Using PRL_L loss...")
            return PrlLCrossEntropyLoss(noise_level=self.hparams.prl_spl_coteaching_noise_level, ignore_index=-1)
        if self.hparams.loss == Losses.SPL_CROSS_ENTROPY:
            logger.info("Using SPL loss...")
            return SplCrossEntropyLoss(noise_level=self.hparams.prl_spl_coteaching_noise_level, ignore_index=-1)
        if self.hparams.loss == Losses.CLIPPED_CROSS_ENTROPY:
            logger.info("Using Clipped CE loss...")
            return ClippedCrossEntropyLoss(
                clip_at_value=self.hparams.cce_clip_loss_at_value,
                start_from_epoch=self.hparams.cce_start_from_epoch,
                ignore_index=-1,
            )
        if self.hparams.loss == Losses.ELR_CROSS_ENTROPY:
            logger.info("Using ELR loss...")
            return ElrCrossEntropyLoss(
                num_observations=self.hparams.num_examples,
                num_classes=self.hparams.num_labels,
                targets_momentum_beta=self.hparams.elr_targets_momentum_beta,
                regularization_constant_lambda=self.hparams.elr_regularization_constant_lambda,
                clamp_margin=self.hparams.elr_clamp_margin,
                ignore_index=-1,
            )
        if self.hparams.loss == Losses.GJSD_LOSS:
            logger.info("Using GJSD loss...")
            return JensenShannonDivergenceLoss(
                num_classes=self.hparams.num_labels,
                num_distributions=self.hparams.gjsd_num_distributions,
                pi_weight=self.hparams.gjsd_pi_weight,
                consistency_regularization=self.hparams.gjsd_consistency_regularization,
            )
        logger.info("Using cross-entropy loss...")
        return CrossEntropyLoss(ignore_index=-1)

    def _get_optimizer(self) -> Optimizer:
        optimizer_grouped_parameters = get_grouped_parameters_with_weight_decay(
            self.named_parameters(), self.hparams.weight_decay
        )
        return AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate)

    def _setup_augmenter(self) -> Augmenter:
        return Augmenter(
            parsed_definitions=self.hparams.gjsd_augmentations, output_length=TokenizerConstants.MAX_LENGTH
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self._encoder(token_ids)

    def configure_optimizers(self) -> T.Tuple[T.List[Optimizer], T.List[T.Dict[str, T.Any]]]:
        optimizer = self._get_optimizer()
        # https://github.com/PyTorchLightning/pytorch-lightning/issues/2056
        scheduler = get_linear_schedule_with_warmup(optimizer, self.hparams.warmup_steps, self.num_training_steps)
        return [optimizer], [{"scheduler": scheduler, "interval": "step"}]

    def _select_labels(self) -> bool:
        return self.hparams.lfnd_logging_enabled and not self.hparams.mixup

    def training_step(self, batch: Batch, batch_index: int) -> T.Dict[str, T.Union[torch.Tensor, int]]:
        token_ids = batch[self._hparams.features_field_name]
        labels = batch[self._hparams.label_field_name]
        logits = self(token_ids)
        probabilities = self._softmax_fn(logits)

        if self.hparams.mixup:
            labels = F.one_hot(labels, num_classes=self._encoder._num_labels)
            # TODO: mixup might be also performed on raw embeddings instead of logits (LFND-40)
            mixup_logit, mixup_label = add_mixup_samples(
                logits, labels, self.hparams.mixup_alpha, self.hparams.mixup_ratio
            )
            labels = torch.vstack((labels, mixup_label))
            logits = torch.vstack((logits, mixup_logit))

        if self.hparams.gjsd_consistency_regularization:
            logits = self.augment_batch(token_ids)

        if self._loss_fn.requires_indices:
            indices = batch[self._hparams.index_field_name]
            loss = self._loss_fn(logits, labels, indices)
        elif self._should_change_to_default_loss():
            _loss_fn = CrossEntropyLoss(ignore_index=-1)
            loss = _loss_fn(logits, labels)
        else:
            loss = self._loss_fn(logits, labels)

        metrics = dict()

        if self.hparams.loss == Losses.ELR_CROSS_ENTROPY:
            loss, cross_entropy_term, elr_regularization = loss
            metrics.update({"elr-ce-term": cross_entropy_term, "elr-regularization": elr_regularization})

        metrics.update({"loss": loss, "epoch": self.current_epoch})
        logger.info(f"Step: {self.global_step}  Loss: {loss}")

        metrics["probabilities"] = probabilities

        if self.hparams.mixup:
            metrics["target"] = batch[self.hparams.label_field_name]
        else:
            metrics["target"] = labels

        if self._select_labels():
            true_label = batch[self._hparams.true_label_field_name]

            if isinstance(self._loss_fn, DropInstancesWithTopValues):
                indices = self._loss_fn.selected_indices
                metrics.update(
                    {
                        BertTrainingConstants.SELECTED_LABELS_KEY: labels[indices],
                        BertTrainingConstants.SELECTED_TRUE_LABELS_KEY: true_label[indices],
                    }
                )

        return metrics

    def augment_batch(self, token_ids: torch.Tensor) -> torch.Tensor:
        augmented_token_batch = [self._augmenter.augment(sample) for sample in token_ids]
        augmented_logit_batch = self(torch.cat(augmented_token_batch, dim=0).to(self._device))
        logits = augmented_logit_batch.reshape(
            token_ids.shape[0], self._augmenter.num_augmentations, self.hparams.num_labels
        )
        return logits.permute(0, 2, 1)

    def validation_step(self, batch: Batch, batch_index: int) -> T.Dict[str, torch.Tensor]:
        token_ids = batch[self.hparams.features_field_name]
        labels = batch[self.hparams.label_field_name]
        logits = self(token_ids)

        if self.hparams.loss == Losses.GJSD_LOSS:
            # Authors of the GJSD paper used Cross Entropy for loss computation on validation samples
            # https://github.com/ErikEnglesson/GJS/blob/1a3c9e0788d4e14194771d096ab5a37ed434cebd/train.py#L343
            _loss_fn = CrossEntropyLoss(ignore_index=-1)
            loss = _loss_fn(logits, labels)
        else:
            loss = self._loss_fn(logits, labels)
        probabilities = self._softmax_fn(logits)

        outputs = {
            "loss": loss,
            "probabilities": probabilities,
            "target": labels,
        }

        if self.hparams.lfnd_logging_enabled:
            true_label = batch[self.hparams.true_label_field_name]
            outputs["true_target"] = true_label

        return outputs

    def _should_change_to_default_loss(self):
        return (
            isinstance(self._loss_fn, ClippedCrossEntropyLoss) and self.current_epoch < self._loss_fn.start_from_epoch
        )

    @staticmethod
    def from_args(args: argparse.Namespace) -> BertClassifier:
        model_args = asdict(BertClassifierParams())
        model_args.update(**vars(args))
        return BertClassifier(**model_args)

    @staticmethod
    def add_argparse_args(
        parent_parser: argparse.ArgumentParser, project_settings: ProjectSettings
    ) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--model-path", type=str, required=True)
        parser.add_argument(
            "--initialize-random-model-weights", dest="initialize_random_model_weights", action="store_true"
        )
        parser.add_argument("--learning-rate", default=BertClassifierParams.learning_rate, type=float)
        parser.add_argument("--warmup-steps", default=BertClassifierParams.warmup_steps, type=int)
        parser.add_argument("--weight-decay", default=BertClassifierParams.weight_decay, type=float)
        parser.add_argument(
            "--features-field-name",
            default=project_settings.features_field_name,
            required=False,
            help=argparse.SUPPRESS,  # used only for internal coherence
        )
        parser.add_argument(
            "--label-field-name",
            default=project_settings.label_field_name,
            required=False,
            help=argparse.SUPPRESS,  # used only for internal coherence
        )
        parser.add_argument(
            "--true-label-field-name",
            default=project_settings.true_label_field_name,
            required=False,
            help=argparse.SUPPRESS,  # used only for internal coherence
        )
        parser.add_argument(
            "--index-field-name",
            default=project_settings.index_field_name,
            required=False,
            help=argparse.SUPPRESS,  # used only for internal coherence
        )
        parser.add_argument("--loss", default=BertClassifierParams.loss, type=str)
        parser.add_argument("--prl-spl-coteaching-noise-level", default=BertClassifierParams.noise_level, type=float)
        parser.add_argument("--elr-targets-momentum-beta", default=0, type=float)
        parser.add_argument("--elr-regularization-constant-lambda", default=0, type=float)
        parser.add_argument("--elr-clamp-margin", default=1e-4, type=float)
        parser.add_argument("--cce-clip-loss-at-value", default=BertTrainingDefaults.CCE_CLIP_VALUE, type=float)
        parser.add_argument("--cce-start-from-epoch", default=BertTrainingDefaults.CCE_START_EPOCH, type=int)
        parser.add_argument("--mixup", action="store_true")
        parser.add_argument("--mixup-alpha", default=BertTrainingDefaults.MIXUP_ALPHA, type=float)
        parser.add_argument("--mixup-ratio", default=BertTrainingDefaults.MIXUP_RATIO, type=float)
        parser.add_argument("--gjsd-num-distributions", default=BertTrainingDefaults.GJSD_NUM_DISTRIBUTIONS, type=int)
        parser.add_argument("--gjsd-pi-weight", default=BertTrainingDefaults.GJSD_PI_WEIGHT, type=float)
        parser.add_argument("--gjsd-consistency-regularization", action="store_true")
        parser.add_argument(
            "--gjsd-augmentations",
            default=BertTrainingDefaults.GJSD_AUGMENTATIONS,
            type=_augmentation_definition_parser,
        )
        parser.add_argument("--lfnd-logging-enabled", action="store_true")

        return parser
