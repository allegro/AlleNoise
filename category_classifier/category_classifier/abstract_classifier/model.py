import logging
import typing as T
from abc import ABC, abstractmethod
from dataclasses import asdict
from functools import partial

import numpy as np
import pytorch_lightning as pl
import sklearn.metrics as sk_metrics
import torch

from category_classifier.bert_classifier.config.constants import BertTrainingConstants
from category_classifier.bert_classifier.config.defaults import BertEvaluationDefaults
from category_classifier.bert_classifier.config.metrics import BaseMetrics, LFNDMetrics
from category_classifier.utils.io_utils import save_json


logger = logging.getLogger(__name__)


Batch = T.Dict[str, torch.Tensor]


class AbstractModel(pl.LightningModule, ABC):
    def __init__(self, *args: T.Any, **kwargs: T.Any) -> None:
        super().__init__()
        self.save_hyperparameters()

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        ...

    @abstractmethod
    def configure_optimizers(self) -> torch.optim.Optimizer:
        ...

    @abstractmethod
    def training_step(self, batch: T.Any, batch_index: int) -> T.Dict[str, T.Union[torch.Tensor, int]]:
        ...

    @abstractmethod
    def validation_step(self, batch: T.Any, batch_index: int) -> T.Dict[str, T.Union[torch.Tensor, int]]:
        ...

    def test_step(self, batch: Batch, batch_index: int) -> T.Dict[str, torch.Tensor]:
        tokens_ids = batch[self.hparams.features_field_name]
        label = batch[self.hparams.label_field_name]
        probabilities = self._softmax_fn(self._encoder(tokens_ids))

        return {"probabilities": probabilities.cpu().detach(), "target": label.cpu().detach()}

    def training_epoch_end(self, outputs: T.List[T.Dict[str, torch.Tensor]]) -> None:
        metrics = self._calculate_metrics_and_dump_predictions_data(outputs, stage="train")

        if BertTrainingConstants.SELECTED_LABELS_KEY in outputs[0]:
            metrics["label_precision"] = self._calculate_label_precision(outputs)

        metrics_with_prefix = self._add_prefix_to_metrics(metrics, "train")
        self.logger.log_metrics(metrics_with_prefix, self.global_step)

        logger.info(f"Evaluation on the training samples")
        logger.info(metrics_with_prefix)

    def validation_epoch_end(self, outputs: T.List[T.Dict[str, torch.Tensor]]) -> T.Dict[str, float]:
        metrics = self._calculate_metrics_and_dump_predictions_data(outputs)
        metrics_with_prefix = self._add_prefix_to_metrics(metrics, "val")

        self.logger.log_metrics(metrics_with_prefix, self.global_step)
        logger.info(f"Validation on a sample of {self._hparams.limit_val_batches} of the whole validation set")
        logger.info(metrics_with_prefix)
        return metrics_with_prefix

    def test_epoch_end(self, outputs: T.List[T.Dict[str, torch.Tensor]]) -> T.Dict[str, float]:
        metrics = self._calculate_metrics_and_dump_predictions_data(
            outputs, output_dir=self._hparams.job_dir, stage="test"
        )
        metrics_with_prefix = self._add_prefix_to_metrics(metrics, "test")

        self.logger.log_metrics(metrics_with_prefix, self.global_step)
        logger.info(metrics_with_prefix)
        return metrics_with_prefix

    def _add_prefix_to_metrics(self, metrics_dictionary: T.Dict[str, float], prefix: str) -> T.Dict[str, float]:
        return {f"{prefix}_{metric_name}": metric_value for metric_name, metric_value in metrics_dictionary.items()}

    def _calculate_metrics_and_dump_predictions_data(
        self, outputs: T.List[T.Dict[str, torch.Tensor]], output_dir: str = None, stage: str = "val"
    ) -> T.Dict[str, float]:

        loss, predictions, targets, true_targets = 0.0, [], [], []

        for output in outputs:
            if len(output["probabilities"]) == 0:
                continue

            predictions.append(output["probabilities"].argmax(dim=1).cpu())
            targets.append(output["target"].cpu())

            if "true_target" in output:
                true_targets.append(output["true_target"].cpu())

            if "loss" in output:
                loss += output["loss"].item()

        if len(predictions) == 0:
            return {}

        concatenated_predictions = np.concatenate(predictions)
        concatenated_targets = np.concatenate(targets)

        metrics = self._calculate_metrics(concatenated_targets, concatenated_predictions)
        metrics = asdict(metrics)

        if "loss" in outputs[0]:
            metrics["loss"] = loss / len(outputs)

        if self._hparams.lfnd_logging_enabled and stage == "val":
            concatenated_true_targets = np.concatenate(true_targets)
            metrics_lfnd = self._calculate_lfnd_metrics(
                concatenated_targets, concatenated_true_targets, concatenated_predictions
            )
            metrics_lfnd = asdict(metrics_lfnd)
            metrics = {**metrics, **metrics_lfnd}

        if output_dir:
            self._save_predictions(outputs, output_dir)

        return metrics

    @staticmethod
    def _calculate_metrics(targets: np.ndarray, predictions: np.ndarray) -> BaseMetrics:
        f1 = partial(sk_metrics.f1_score, targets, predictions, zero_division=0)
        precision = partial(sk_metrics.precision_score, targets, predictions, zero_division=0)
        recall = partial(sk_metrics.recall_score, targets, predictions, zero_division=0)

        metrics = BaseMetrics(
            accuracy_score=sk_metrics.accuracy_score(targets, predictions),
            precision_macro=precision(average="macro"),
            recall_macro=recall(average="macro"),
            f1_score_macro=f1(average="macro"),
        )

        return metrics

    @staticmethod
    def _calculate_lfnd_metrics(targets: np.ndarray, true_targets: np.ndarray, predictions: np.ndarray) -> LFNDMetrics:

        true_samples_mask = targets == true_targets

        accuracy_clean = sk_metrics.accuracy_score(true_targets[true_samples_mask], predictions[true_samples_mask])
        accuracy_noisy = sk_metrics.accuracy_score(true_targets[~true_samples_mask], predictions[~true_samples_mask])
        memorized_noisy = sk_metrics.accuracy_score(targets[~true_samples_mask], predictions[~true_samples_mask])

        lfnd_metrics = LFNDMetrics(
            fraction_clean_correct=accuracy_clean,
            fraction_clean_incorrect=1 - accuracy_clean,
            fraction_noisy_correct=accuracy_noisy,
            fraction_noisy_incorrect=1 - accuracy_noisy,
            fraction_noisy_memorized=memorized_noisy,
        )

        return lfnd_metrics

    @staticmethod
    def _calculate_label_precision(outputs: T.List[T.Dict[str, torch.Tensor]]) -> float:
        selected_labels = [output[BertTrainingConstants.SELECTED_LABELS_KEY].cpu() for output in outputs]
        selected_true_labels = [output[BertTrainingConstants.SELECTED_TRUE_LABELS_KEY].cpu() for output in outputs]

        concatenated_labels = np.concatenate(selected_labels)
        concatenated_true_labels = np.concatenate(selected_true_labels)
        label_precision = np.mean(concatenated_labels == concatenated_true_labels)
        return label_precision

    @staticmethod
    def _save_predictions(outputs: T.List[T.Dict[str, torch.Tensor]], output_dir: str) -> None:

        predictions_data = [
            {
                "values": output["probabilities"].max(dim=1).values.tolist(),
                "indices": output["probabilities"].argmax(dim=1).tolist(),
                "targets": output["target"].tolist(),
            }
            for output in outputs
        ]

        save_json(predictions_data, output_dir=output_dir, output_filename=BertEvaluationDefaults.PREDICTIONS_FILE_NAME)

    def _get_num_batches_per_epoch(self) -> int:
        limit_batches = self.trainer.limit_train_batches
        batches = len(self.trainer.datamodule.train_dataloader())
        batches = min(batches, limit_batches) if isinstance(limit_batches, int) else int(limit_batches * batches)

        if self.hparams.train_distributed:
            effective_accum = self.trainer.accumulate_grad_batches
        else:
            num_devices = max(1, self.trainer.num_gpus, self.trainer.num_processes)

            if self.trainer.tpu_cores:
                num_devices = max(num_devices, self.trainer.tpu_cores)
            effective_accum = self.trainer.accumulate_grad_batches * num_devices

        num_batches = batches // effective_accum

        return num_batches

    @property
    def num_training_steps(self) -> int:
        """Total training steps inferred from datamodule and devices."""
        if self._num_training_steps != -1:
            return self._num_training_steps

        if self.trainer.max_steps != -1:
            self._num_training_steps = self.trainer.max_steps
            return self.trainer.max_steps

        num_batches_per_epoch = self._get_num_batches_per_epoch()
        self._num_training_steps = num_batches_per_epoch * self.trainer.max_epochs
        logger.info(f"Training steps: {self._num_training_steps} = {num_batches_per_epoch} * {self.trainer.max_epochs}")

        return self._num_training_steps
