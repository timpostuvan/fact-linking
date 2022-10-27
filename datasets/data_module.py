from typing import Any

import torch
from omegaconf import DictConfig
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from .batch_sample import BatchedSample
from .dataset import ComFactDataset


class ComFactDataModule(LightningDataModule):
    def __init__(self, config: DictConfig):
        super().__init__()

        self.config = config
        self.train_set, self.dev_set, self.test_set = "train", "dev", "test"
        self.dataset_config, self.training_config = config.data, config.training
        self.encoder_config, self.decoder_config = config.model.encoder, config.model.decoder
        self.optimizer_config = config.optimization

        self.datasets = {}

    def create_dataset(self, stage: str) -> Any:
        if stage == "train":
            return ComFactDataset(
                dataset_path=self.dataset_config.train_path,
                batch_size=self.dataset_config.batch_size,
                model_name=self.encoder_config.name,
                max_node_num=self.decoder_config.max_node_num,
                max_seq_length=self.training_config.max_seq_len
            )
        elif stage == "dev":
            return ComFactDataset(
                dataset_path=self.dataset_config.dev_path,
                batch_size=self.dataset_config.batch_size,
                model_name=self.encoder_config.name,
                max_node_num=self.decoder_config.max_node_num,
                max_seq_length=self.training_config.max_seq_len
            )
        elif stage == "test":
            return ComFactDataset(
                dataset_path=self.dataset_config.test_path,
                batch_size=self.dataset_config.batch_size,
                model_name=self.encoder_config.name,
                max_node_num=self.decoder_config.max_node_num,
                max_seq_length=self.training_config.max_seq_len
            )
        else:
            raise ValueError(f"Unknown stage {stage}")

    def setup(self, stage: str = None):
        assert stage is not None

        stages = []
        if stage == "fit":
            stages += [self.train_set, self.dev_set]
        if stage == "test":
            stages += [self.test_set]

        for _stage in stages:
            self.datasets[_stage] = self.create_dataset(stage=_stage)

    def train_dataloader(self):
        return DataLoader(
            dataset=self.datasets[self.train_set],
            batch_size=self.dataset_config.batch_size,
            collate_fn=self._collate,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.datasets[self.dev_set],
            batch_size=self.dataset_config.batch_size,
            collate_fn=self._collate,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.datasets[self.test_set],
            batch_size=self.dataset_config.batch_size,
            collate_fn=self._collate,
            shuffle=False,
        )

    @staticmethod
    def _collate(batch: Any) -> Any:
        batch = BatchedSample(batch)
        batch.to_tensors()
        return batch

    def transfer_batch_to_device(self, batch: BatchedSample, device: torch.device, dataloader_idx: int):
        batch.to_device(device)
        return batch
