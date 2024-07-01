import typing as T
from logging import Logger

import pytorch_lightning as pl

import utils.py_lighting_utils as pl_utils


def setup_learning_rate_logger_callback() -> pl.callbacks.Callback:
    return pl.callbacks.LearningRateMonitor()


def setup_checkpoint_trainer_callback(
    save_frequency: float, num_epochs: int, ckpt_path: str, logger: Logger
) -> pl.callbacks.Callback:
    return pl_utils.CheckpointEveryNEpochs(
        save_frequency=save_frequency,
        checkpoint_path=ckpt_path,
        num_epochs=num_epochs,
        logger=logger,
    )


def setup_fine_selector_callback(fine_threshold: float, normalize: bool, fine_batch_size: int) -> pl.callbacks.Callback:
    return pl_utils.FineSelector(fine_threshold, normalize, fine_batch_size)


def setup_callbacks(
    job_dir: str,
    logger: Logger,
    checkpoint_save_frequency: float,
    num_epochs: int,
) -> T.List[pl.callbacks.Callback]:
    callbacks = [
        setup_learning_rate_logger_callback(),
        setup_checkpoint_trainer_callback(checkpoint_save_frequency, num_epochs, job_dir, logger),
    ]
    return callbacks
