#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

import hydra
import lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig
from torch.utils.data import Dataset, DistributedSampler

from uni2ts.common import hydra_util  # noqa: hydra resolvers
from uni2ts.data.builder import DatasetBuilder


class DataModule(L.LightningDataModule):
    def __init__(self, cfg: DictConfig, dataset: Dataset):
        super().__init__()
        self.cfg = cfg
        self.dataset = dataset

    def train_dataloader(self):
        sampler = (
            DistributedSampler(
                self.dataset,
                num_replicas=None,
                rank=None,
                shuffle=self.cfg.train_dataloader.shuffle,
                seed=0,
                drop_last=False,
            )
            if self.trainer.world_size > 1
            else None
        )
        train_dataloader = instantiate(self.cfg.train_dataloader, _partial_=True)(
            dataset=self.dataset,
            shuffle=self.cfg.train_dataloader.shuffle if sampler is None else None,
            sampler=sampler,
            batch_size=self.batch_size,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )
        return train_dataloader

    @property
    def batch_size(self) -> int:
        return self.cfg.train_dataloader.batch_size // (
            self.trainer.world_size * self.trainer.accumulate_grad_batches
        )

    @property
    def num_batches_per_epoch(self) -> int:
        return (
            self.cfg.train_dataloader.num_batches_per_epoch
            * self.trainer.accumulate_grad_batches
        )


@hydra.main(version_base="1.3", config_path="conf/pretrain", config_name="default")
def main(cfg: DictConfig):
    if cfg.tf32:
        assert cfg.trainer.precision == 32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    model: L.LightningModule = instantiate(cfg.model, _convert_="all")
    if cfg.compile:
        model.module.compile(mode=cfg.compile)
    trainer: L.Trainer = instantiate(cfg.trainer)
    dataset_builder: DatasetBuilder = instantiate(cfg.data)
    dataset: Dataset = dataset_builder.load_dataset(model.create_transform_map())
    L.seed_everything(cfg.seed + trainer.logger.version, workers=True)
    trainer.fit(
        model,
        train_dataloaders=DataModule(cfg, dataset),
        ckpt_path="last",
    )


if __name__ == "__main__":
    main()
