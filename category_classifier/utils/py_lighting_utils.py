import math
import os
import typing as T
from abc import ABC
from argparse import Namespace
from logging import Logger

import numpy as np
import pytorch_lightning as pl
import torch
import yaml
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities import rank_zero_only
from torch import Tensor
from torch.utils.tensorboard import SummaryWriter

from abstract_classifier.model import AbstractModel
from bert_classifier.config.constants import BertTrainingConstants
from utils import io_utils as io


def get_grouped_parameters_with_weight_decay(
    named_parameters: T.Iterator[T.Tuple[str, Tensor]], weight_decay: float
) -> T.List[dict]:
    no_decay_param_names = ["bias", "LayerNorm.weight"]
    decay_params = []
    no_decay_params = []

    for param_name, param in named_parameters:
        if any(no_decay_param_name in param_name for no_decay_param_name in no_decay_param_names):
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]


class TensorBoardLogger(pl.loggers.TensorBoardLogger):
    """Wrapper on Lightning's TensorBoardLogger that is able to deal with Google
    Cloud Storage and removes this weird subfolder structure
    (e.g. `version_0/default`) that is added to `save_dir` by Lightning.
    """

    def __init__(self, save_dir: str, **kwargs) -> None:
        super().__init__(save_dir, name="", version=None, **kwargs)

    @rank_zero_only
    def log_hyperparams(
        self,
        params: T.Union[T.Dict[str, T.Any], Namespace],
        metrics: T.Optional[T.Dict[str, T.Any]] = None,
    ) -> None:

        params = self._convert_params(params)
        log_params = params.copy()
        log_params.pop("config", None)

        super().log_hyperparams(log_params, metrics)

    @property
    def experiment(self) -> SummaryWriter:
        if self._experiment is not None:  # type: ignore
            return self._experiment  # type: ignore

        if not io.isdir(self.root_dir):
            io.makedirs(self.root_dir)

        self._experiment = SummaryWriter(log_dir=self.log_dir, **self._kwargs)
        return self._experiment

    @rank_zero_only
    def log_metrics(self, metrics: T.Dict[str, T.Any], step: T.Optional[T.Union[int, float]] = None) -> None:
        for metric_name, metric_value in metrics.items():
            try:
                self.experiment.add_scalar(metric_name, metric_value, step)
            except NotImplementedError:
                self.experiment.add_scalars(metric_name, metric_value, step)

    @rank_zero_only
    def save(self) -> None:
        pl.loggers.base.LightningLoggerBase.save(self)

        self.experiment.flush()

        dir_path = self.log_dir
        if not io.isdir(dir_path):
            dir_path = self.save_dir

        hparams_file = os.path.join(dir_path, self.NAME_HPARAMS_FILE)

        if io.file_exists(hparams_file):
            return

        hparams_str = {key: str(value) for key, value in self.hparams.items()}
        save_hparams_to_yaml(hparams_file, hparams_str)

    @property
    def version(self) -> str:
        # We neeed to return an empty string here, b/c otherwise
        # configure_checkpoint_callback (in TrainerCallbackConfigMixin) will
        # blow up.
        return ""


def save_hparams_to_yaml(config_yaml: str, hparams: T.Union[dict, Namespace]) -> None:
    """
    Utility method to save all Lightning (and not only) hparams to one unique yaml file
    """

    check_saving_dir(config_yaml)

    if isinstance(hparams, Namespace):
        hparams = vars(hparams)

    corrected_hparams = correct_ambiguous_hparams(hparams)

    with io.open_file(config_yaml, mode="w") as fp:
        yaml.dump(corrected_hparams, fp)


def check_saving_dir(config_yaml: str) -> None:
    if not config_yaml.startswith("gs://") and not os.path.isdir(os.path.dirname(config_yaml)):
        raise RuntimeError(f"Missing folder: {os.path.dirname(config_yaml)}.")


def correct_ambiguous_hparams(hparams: T.Union[dict, Namespace]) -> T.Union[dict, Namespace]:
    # remove ``checkpoint_callback`` and ``logger``
    # otherwise won't dump the file (pickle error)
    if "checkpoint_callback" in hparams:
        hparams["checkpoint_callback"] = None
    if "logger" in hparams:
        hparams["logger"] = None
    # we need to hard-code this to avoid TypeError during yaml dumping
    if "entity_type" in hparams:
        hparams["entity_type"] = str(hparams["entity_type"])
    return hparams


class AbstractCheckpoint(Callback, ABC):
    """
    Abstract class for checkpoint callback
    """

    def __init__(self, save_frequency: float, checkpoint_path: str, logger: Logger) -> None:
        self._save_frequency = save_frequency
        self._checkpoint_path = checkpoint_path
        self._logger = logger

    @staticmethod
    def save_pytorch_lightning_checkpoint(
        trainer: pl.Trainer,
        ckpt_path: str,
        logger: Logger,
        weights_only: bool = False,
    ) -> None:
        """
        Utility method that saves pytorch-lightning checkpoint (useful for resuming training) to
        the specified path
        """

        ckpt_dir = os.path.dirname(ckpt_path)
        if not io.isdir(ckpt_dir):
            io.makedirs(ckpt_dir)
        logger.info(f"Dumping checkpoint to {ckpt_path}")
        trainer.save_checkpoint(ckpt_path, weights_only=weights_only)

    @staticmethod
    def save_encoder(trainer: pl.Trainer, encoder_dir: str, logger: Logger) -> None:
        """
        Utility method that saves only encoder weights and configs (useful for testing/production) to
        the specified path
        """
        local_encoder_dir = os.path.join(
            "/tmp", BertTrainingConstants.CHECKPOINT_DIRNAME, os.path.basename(encoder_dir)
        )
        if not io.isdir(local_encoder_dir):
            io.makedirs(local_encoder_dir)
        logger.info(f"Dumping encoder weights to {local_encoder_dir}")
        encoder_to_dump = trainer.lightning_module._encoder
        encoder_to_dump.dump_encoder_weights(local_encoder_dir, "encoder.pt")
        io.copy_dir(local_encoder_dir, encoder_dir)


class CheckpointEveryNEpochs(AbstractCheckpoint):
    def __init__(
        self,
        num_epochs: int,
        checkpoint_path: str,
        save_frequency: float,
        logger: Logger,
    ) -> None:
        super().__init__(checkpoint_path=checkpoint_path, save_frequency=save_frequency, logger=logger)
        self._num_epochs = num_epochs
        self._num_steps_per_epoch = 0
        self._num_steps_after_prev_epoch = 0
        self._last_step = 0

    def on_train_epoch_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self._num_steps_per_epoch = pl_module._get_num_batches_per_epoch()
        self._num_steps_after_prev_epoch = trainer.global_step

        if trainer.current_epoch == 0:
            self._last_step = pl_module.num_training_steps - 1
        elif trainer.current_epoch + 1 == self._num_epochs:
            self._last_step = self._num_steps_after_prev_epoch + self._num_steps_per_epoch - 1

    def on_train_batch_end(
        self,
        trainer: pl.Trainer,
        pl_module: pl.LightningModule,
        outputs: T.Any,
        batch: T.Any,
        batch_idx: int,
        unused: T.Optional[int] = 0,
    ) -> None:
        """Check if we should save a checkpoint after every train batch"""
        if trainer.global_rank != 0:
            return

        global_step = trainer.global_step
        epoch_step = global_step - self._num_steps_after_prev_epoch
        updated_current_epoch = trainer.current_epoch + (epoch_step + 1) / self._num_steps_per_epoch

        num_checkpoints_per_epoch = int(1 / self._save_frequency)
        num_steps_between_checkpoints = (self._num_steps_per_epoch) / (num_checkpoints_per_epoch)
        checkpoint_steps = np.array(
            [
                np.ceil(self._num_steps_after_prev_epoch + num_steps_between_checkpoints * checkpoint_index) - 1
                for checkpoint_index in range(1, num_checkpoints_per_epoch + 1)
            ]
        )

        if global_step in checkpoint_steps:
            if global_step == self._last_step:
                encoder_name = BertTrainingConstants.FINAL_PRODUCTION_CHECKPOINT_NAME
                ckpt_name = BertTrainingConstants.FINAL_CHECKPOINT_FILENAME
            else:
                encoder_name = BertTrainingConstants.production_checkpoint_name(updated_current_epoch)
                ckpt_name = BertTrainingConstants.LATEST_CHECKPOINT_FILENAME
            encoder_dir = os.path.join(self._checkpoint_path, BertTrainingConstants.CHECKPOINT_DIRNAME, encoder_name)
            ckpt_path = os.path.join(self._checkpoint_path, BertTrainingConstants.CHECKPOINT_DIRNAME, ckpt_name)

            self.save_pytorch_lightning_checkpoint(trainer, ckpt_path, self._logger)
            self.save_encoder(trainer, encoder_dir, self._logger)


class FineSelector(pl.Callback):
    def __init__(self, threshold: float, normalize: bool, batch_size: int) -> None:
        """
        Callback that is responsible for executing FINE method at the end of each epoch (except of the last one).

        :param threshold: threshold used for FINE method (threshold of clean probability)
        :param normalize: bool parameter for FINE method (should the features be normalized)
        :param batch_size: maximum size of the batch for FINE processing (to avoid out of memory error)
        """
        super().__init__()
        self.threshold = threshold
        self.normalize = normalize
        self.batch_size = batch_size

    def _update_dataset(self, trainer: pl.Trainer, model: AbstractModel):
        """Update the dataset so it contains only samples selected by FINE."""

        # move all tensors to the same device
        device = "cuda" if torch.cuda.is_available() else "cpu"
        token_ids = torch.tensor(trainer.datamodule._train_dataset_complete.token_ids, dtype=torch.int32).to(device)
        labels = torch.tensor(trainer.datamodule._train_dataset_complete.labels, dtype=torch.int32).to(device)
        indices = torch.arange(len(labels)).to(device)

        # indices of data points that should be selected after performing FINE
        all_selected_idx = []

        # split data into smaller data chunks (according to their labels)
        label_chunks = self.generate_label_chunks(labels)
        for label_chunk in label_chunks:
            # perform this only to the data points from selected labels
            tokens_in_chunk, labels_in_chunk, indices_in_chunk = self.filter_by_labels(
                torch.Tensor(label_chunk).to(device), token_ids, labels, indices
            )
            # split the chunk into smaller batches if it is still bigger than the batch size
            # (it may happen for the big classes)
            tokens_batches, labels_batches, indices_batches = self.split_to_batches(
                tokens_in_chunk, labels_in_chunk, indices_in_chunk
            )
            for tokens_batch, labels_batch, indices_batch in zip(tokens_batches, labels_batches, indices_batches):
                # get the representation of the offer title
                features = model._encoder.mean_activation_of_the_last_hidden_state(tokens_batch)
                # perform FINE sample selection
                selected_idx = fine_dataset_split(features, labels_batch, self.threshold, self.normalize, indices_batch)
                # append indices of selected data points to the all_selected_idx
                all_selected_idx.extend(selected_idx)

        # overwrite the train dataset so it is limited only to the selected data points
        # dataloader will be reloaded based on this new dataset
        trainer.datamodule._set_train_subset(all_selected_idx)

    def on_train_epoch_end(self, trainer: pl.Trainer, model: AbstractModel, **kwargs):
        """
        This callback will be executed at the end of the epoch (after the validation).
        It has access to the dataset - token_ids, labels (in trainer.datamodule) and to the encoder (in model._encoder).
        """
        if trainer.current_epoch + 1 < trainer.max_epochs:
            model.eval()
            self._update_dataset(trainer, model)

    def generate_label_chunks(self, labels: torch.Tensor) -> T.List[T.List[int]]:
        """
        Group labels together so they generate chunks of size = fine_batch_size.
        This method prevents from dividing one class into multiple chunks.
        The only exception when chunk size > batch size is when the single class size is bigger than batch size.

        Example:
            batch_size = 3
            labels = [2,6,3,2,1,6,3,4,3,6,5,6,4]
            labels_sorted = [1,2,2,3,3,3,4,4,5,6,6,6,6]
            unique_labels = [1,2,3,4,5,6]

            label_chunks:
            [1,2] - in total 3 data points
            [3] - in total 3 data points
            [4,5] - in total 3 data points
            [6] - in total 4 data points, but from the same class

            output:
            [[1.0, 2.0], [3.0], [6.0], [4.0, 5.0]]
                - label chunk [6.0] is reported before [4.0, 5.0] because it skips part of the logic
                (with appending the result to labels_in_chunk) and directly creates new chunk in label_chunks
        """
        chunk_size = 0
        label_chunks = []
        labels_in_chunk = []

        for label in labels.unique():
            class_size = len(labels[labels == label])
            if class_size > self.batch_size:
                label_chunks.append([label.item()])
            elif chunk_size + class_size <= self.batch_size:
                labels_in_chunk.append(label.item())
                chunk_size += class_size
            else:
                label_chunks.append(labels_in_chunk)
                labels_in_chunk = [label.item()]
                chunk_size = class_size

        label_chunks.append(labels_in_chunk)
        return label_chunks

    def filter_by_labels(
        self, labels_in_chunk: torch.Tensor, token_ids: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Filter token_ids, labels and indices only to the data points from the classes included in labels_in_chunk."""
        cond = torch.isin(labels, labels_in_chunk)
        tokens_in_chunk = token_ids[cond]
        labels_in_chunk = labels[cond]
        indices_in_chunk = indices[cond]
        return tokens_in_chunk, labels_in_chunk, indices_in_chunk

    def split_to_batches(
        self, token_ids: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor
    ) -> T.Tuple[T.Tuple[torch.Tensor], T.Tuple[torch.Tensor], T.Tuple[torch.Tensor]]:
        """
        In case when chunk size is higher than batch size, this method divides data into batches.
        It is used with single-class chunks, when class size > batch size.
        """
        num_batches = max(math.ceil(len(labels) / self.batch_size), 1)
        tokens_batches = torch.tensor_split(token_ids, num_batches)
        labels_batches = torch.tensor_split(labels, num_batches)
        indices_batches = torch.tensor_split(indices, num_batches)
        return tokens_batches, labels_batches, indices_batches


def look_for_checkpoints_in_job_dir(job_dir: str) -> T.Union[str, None]:
    checkpoint_path = os.path.join(job_dir, "checkpoints", BertTrainingConstants.LATEST_CHECKPOINT_FILENAME)
    return checkpoint_path if io.file_exists(checkpoint_path) else None
