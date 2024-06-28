import argparse
import logging
import warnings
from dataclasses import asdict

from pytorch_lightning import seed_everything

from bert_classifier.config import callbacks, infrastructure
from bert_classifier.config.defaults import TrainerLoggingConfig
from bert_classifier.data.dataset import BertDataModule, AnchorTrainingProjectSettings
from bert_classifier.data.utils import (
    check_data_consistency_and_determine_num_labels_and_num_examples,
    determine_num_steps_per_epoch,
)
from bert_classifier.model.co_teaching_model import (
    CoTeachingBertClassifier,
    CoTeachingBertClassifierParams,
)
from bert_classifier.model.model import BertClassifier
from bert_classifier.project_setting import ProjectSettings
from bert_classifier.trainer.trainer import Trainer
from utils import py_lighting_utils
from utils.py_lighting_utils import look_for_checkpoints_in_job_dir


warnings.filterwarnings("ignore", message="This overload of add_ is deprecated", category=UserWarning)

logger = logging.getLogger(__name__)


class TrainingJob:
    def __init__(self, project_setting: ProjectSettings) -> None:
        self._project_settings = project_setting
        self._parser = self._add_parser()

    def _add_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser()

        # data config
        parser = BertDataModule.add_argparse_args(parser, self._project_settings)

        # training config
        parser = Trainer.add_argparse_args(parser, self._project_settings)
        parser = BertClassifier.add_argparse_args(parser, self._project_settings)
        parser = CoTeachingBertClassifier.add_argparse_args(parser)

        parser.add_argument("--seed", default=42, type=int)
        parser.add_argument("--job-dir", type=str, required=True)

        return parser

    def _setup_trainer(self, args: argparse.Namespace) -> Trainer:
        args = self._extend_args_with_internal_configs(args)
        logger.info(args)
        tb_logger = py_lighting_utils.TensorBoardLogger(save_dir=args.job_dir, flush_secs=60)
        all_callbacks = callbacks.setup_callbacks(
            logger=logger,
            job_dir=args.job_dir,
            checkpoint_save_frequency=args.checkpoint_save_frequency_fraction,
            num_epochs=args.num_epochs,
        )

        trainer = Trainer.from_argparse_args(args, logger=tb_logger, callbacks=all_callbacks)
        return trainer

    def _extend_args_with_internal_configs(self, args: argparse.Namespace) -> argparse.Namespace:
        num_labels, num_examples = check_data_consistency_and_determine_num_labels_and_num_examples(
            args.train_file_path, self._project_settings
        )
        vars(args).update({"num_labels": num_labels, "num_examples": num_examples})

        train_infra_conf = (
            infrastructure.InfrastructureConfig.DDP_CONFIG_DEFAULT
            if args.train_distributed
            else infrastructure.InfrastructureConfig.SINGLE_MACHINE_CONFIG_DEFAULT
        )
        world_size = infrastructure.compute_world_size(args.gpus, args.num_nodes)
        num_steps_per_epoch = determine_num_steps_per_epoch(args.batch_size, world_size, args.num_examples)
        logger.info(f"Training infrastructure config: {train_infra_conf}")
        logger.info(f"World size: {world_size}")
        logger.info(f"Number of steps required to complete an epoch: {num_steps_per_epoch}")
        logger.info(f"Amp config: {infrastructure.AmpConfig.FP_OPT_DEFAULT}")

        # TODO use parameter names defined by ptl in scripts
        other_config = {
            "limit_val_batches": args.validation_sample_size,
            "resume_from_checkpoint": args.pre_trained_model_checkpoint,
            "max_epochs": args.num_epochs,
            "num_steps_per_epoch": num_steps_per_epoch,
            "world_size": world_size,
            "reload_dataloaders_every_n_epochs": args.reload_dataloaders_every_n_epochs,
        }
        for config in [
            asdict(TrainerLoggingConfig()),
            infrastructure.AmpConfig.FP_OPT_DEFAULT,
            train_infra_conf,
            other_config,
        ]:
            vars(args).update(config)
        return args

    def _setup_data_module(self, args: argparse.Namespace) -> BertDataModule:
        return BertDataModule(args)

    def run(self) -> None:
        args = self._parser.parse_args()

        seed_everything(args.seed)
        logger.info("Setting up the Trainer obj...")
        trainer = self._setup_trainer(args)

        logger.info("Preparing the data...")
        data_module = self._setup_data_module(args)
        logger.info(data_module.dataset_type)

        logger.info("Setting up the Model obj...")
        if not args.co_teaching:
            model = BertClassifier.from_args(args)
        else:
            args.loss = CoTeachingBertClassifierParams.loss
            model = CoTeachingBertClassifier.from_args(args)

        checkpoint_path = look_for_checkpoints_in_job_dir(args.job_dir)
        if checkpoint_path:
            logger.info(f"Resuming training from: {checkpoint_path}")

        logger.info("Training started")
        trainer.fit(model, data_module, ckpt_path=checkpoint_path)
        if args.test_file_path != "":
            logger.info("Testing started")
            trainer.test(model, data_module)


if __name__ == "__main__":
    training_job = TrainingJob(AnchorTrainingProjectSettings())
    training_job.run()
