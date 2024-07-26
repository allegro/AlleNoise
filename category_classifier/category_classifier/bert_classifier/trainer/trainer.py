import argparse

import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_warn

from category_classifier.bert_classifier.config.defaults import BertTrainingDefaults
from category_classifier.bert_classifier.project_setting import ProjectSettings
from category_classifier.utils import io_utils as io


class Trainer(pl.Trainer):
    def save_checkpoint(self, filepath: str, weights_only: bool = False) -> None:
        if not hasattr(self, "checkpoint_connector"):
            return

        checkpoint = self.checkpoint_connector.dump_checkpoint(weights_only)

        if self.is_global_zero:
            try:
                io.torch_save_state_dict(checkpoint, filepath)
            except AttributeError as err:
                if pl.LightningModule.CHECKPOINT_HYPER_PARAMS_KEY in checkpoint:
                    del checkpoint[pl.LightningModule.CHECKPOINT_HYPER_PARAMS_KEY]
                rank_zero_warn(
                    "Warning, `module_arguments` dropped from checkpoint." f" An attribute is not picklable {err}"
                )
                io.torch_save_state_dict(checkpoint, filepath)

    @staticmethod
    def add_argparse_args(
        parser: argparse.ArgumentParser, project_settings: ProjectSettings
    ) -> argparse.ArgumentParser:
        parser = pl.Trainer.add_argparse_args(parser)
        parser.add_argument(
            "--checkpoint-save-frequency-fraction",
            type=float,
            default=BertTrainingDefaults.CHECKPOINT_SAVE_FREQUENCY_FRACTION,
            help="Define how often to save a checkpoint within an epoch."
            " E.g, if set to 0.5 checkpoint saving will be triggered every half of epoch",
        )
        parser.add_argument(
            "--pre-trained-model-checkpoint",
            type=str,
            default=None,
            help="Path to .ckpt file - ignored in case resume_training is not used as arg",
        )
        parser.add_argument(
            "--stage",
            type=str,
            default=BertTrainingDefaults.TRAIN_STAGE_NAME,
            help="Specifies the stage of the Trainer obj.",
        )
        parser.add_argument(
            "--num-epochs",
            type=int,
            default=BertTrainingDefaults.NUM_EPOCHS,
            help="Specify number of training epochs",
        )
        parser.add_argument(
            "--validation-sample-size",
            type=float,
            default=BertTrainingDefaults.VALID_SAMPLE_SIZE,
            help="Specifies the sample size (e.g percentage of validation batches over the total)"
            "used for validation. Set to 0.0 to completely disable validation",
        )
        parser.add_argument(
            "--reload-dataloaders-every-n-epochs",
            type=int,
            default=BertTrainingDefaults.RELOAD_DATALOADERS,
            help="Define if dataloaders should be reloaded every epoch. Set to True for FINE",
        )
        parser.add_argument(
            "--train-distributed",
            action="store_true",
            help="Enable distributed training with Distributed Data Parallel",
        )
        parser.add_argument(
            "--entity",
            default=project_settings.entity,
            required=False,
            help=argparse.SUPPRESS,  # used only for internal coherence
        )
        return parser
