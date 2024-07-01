import typing as T
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace

import pytorch_lightning as pl
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset

from bert_classifier.config.defaults import BertTrainingDefaults
from utils.data_utils import seed_worker


class AbstractDataModule(pl.LightningDataModule, ABC):
    def __init__(self, hparams: Namespace) -> None:
        super().__init__()
        self._hparams = hparams

        self._train_dataset = None
        self._val_dataset = None
        self._test_dataset = None

    @abstractmethod
    def prepare_data(self) -> None:
        ...

    @abstractmethod
    def setup(self, stage: str = None) -> None:
        ...

    def create_dataloader(
        self,
        dataset: Dataset,
        shuffle: bool = True,
        collate_fn: T.Optional[T.Callable] = None,
    ) -> DataLoader:
        deterministic_args = self._get_deterministic_args()
        return DataLoader(
            dataset,
            batch_size=self._hparams.batch_size,
            num_workers=self._hparams.num_workers,
            collate_fn=collate_fn,
            shuffle=shuffle,
            pin_memory=False,
            **deterministic_args,
        )

    def train_dataloader(self) -> DataLoader:
        return self.create_dataloader(self._train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self.create_dataloader(self._val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self.create_dataloader(self._test_dataset, shuffle=False)

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch-size", default=BertTrainingDefaults.BATCH_SIZE, type=int)
        parser.add_argument("--num-workers", default=BertTrainingDefaults.NUM_WORKERS, type=int)

        return parser

    def _get_deterministic_args(self) -> T.Dict[str, T.Any]:
        if self._hparams.deterministic:
            random_generator = torch.Generator()
            random_generator.manual_seed(self._hparams.seed)
            return {
                "worker_init_fn": seed_worker,
                "generator": random_generator,
            }
        else:
            return {}
