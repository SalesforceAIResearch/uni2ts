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

from typing import Optional

import hydra
import lightning as L
import torch
from hydra.utils import call, get_class, instantiate
from omegaconf import DictConfig
from torch.utils.data import Dataset, DistributedSampler

from uni2ts.common import hydra_util  # noqa: hydra resolvers
import sys


# class Logger(object):
#     def __init__(self, filename="Default.log"):
#         self.terminal = sys.stdout
#         self.log = open(filename, "a")
#         self.encoding = None
#
#     def write(self, message):
#         self.terminal.write(message)
#         self.log.write(message)
#
#     def flush(self):
#         pass

class DataModule(L.LightningDataModule):
    def __init__(
        self, cfg: DictConfig, train_dataset: Dataset, val_dataset: Optional[Dataset]
    ):
        super().__init__()
        self.cfg = cfg
        self.train_dataset = train_dataset

        if val_dataset is not None:
            self.val_dataset = val_dataset
            self.val_dataloader = self._val_dataloader

    def train_dataloader(self):
        sampler = (
            DistributedSampler(
                self.train_dataset,
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
            dataset=self.train_dataset,
            shuffle=self.cfg.train_dataloader.shuffle if sampler is None else None,
            sampler=sampler,
            batch_size=self.batch_size,
            num_batches_per_epoch=self.num_batches_per_epoch,
        )
        return train_dataloader

    def _val_dataloader(self):
        sampler = (
            DistributedSampler(
                self.val_dataset,
                num_replicas=None,
                rank=None,
                shuffle=self.cfg.val_dataloader.shuffle,
                seed=0,
                drop_last=False,
            )
            if self.trainer.world_size > 1
            else None
        )
        val_dataloader = instantiate(self.cfg.val_dataloader, _partial_=True)(
            dataset=self.val_dataset,
            shuffle=self.cfg.val_dataloader.shuffle if sampler is None else None,
            sampler=sampler,
            batch_size=self.batch_size,
        )
        return val_dataloader

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


@hydra.main(version_base="1.3", config_path="conf/finetune", config_name="default")
def main(cfg: DictConfig):
    if cfg.tf32:
        assert cfg.trainer.precision == 32
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # Methods to initiate objects in hydra:
    # https://hydra.cc/docs/advanced/instantiate_objects/overview/
    # `get_class`: Look up a class based on a dotpath.
    # `instantiate` for creating objects and `call` for invoking functions
    model: L.LightningModule = get_class(cfg.model._target_).load_from_checkpoint(
        **call(cfg.model._args_, _convert_="all"),
    )
    if cfg.compile:
        model.module.compile(mode=cfg.compile)
    trainer: L.Trainer = instantiate(cfg.trainer)

    # cfg.data: use the corresponding yaml in data folder based on the passed data name.
    # train_dataset only contains the range before offset.
    train_dataset: Dataset = instantiate(cfg.data).load_dataset(
        model.create_train_transform()  # This transform includes patching and flatten,etc
    )

    # val data in disk contains the whole range of data, including train, val and test.
    # Specify offset, eval_length, etc in yaml when loading.
    # val_dataset is a ConcatDataset, with various config for transformation.
    val_dataset: Optional[Dataset] = (
        instantiate(cfg.val_data).load_dataset(model.create_val_transform)
        if "val_data" in cfg
        else None
    )
    L.seed_everything(cfg.seed + trainer.logger.version, workers=True)


    # log_dir = cfg.trainer.logger.save_dir + '/' + 'log.txt'
    # sys.stdout = Logger(log_dir)
    #
    # print(f'log version: {trainer.logger.version}')
    # print(f'Training batch size: {cfg.train_dataloader.batch_size}')
    # print(f'Num batches per epoch: {cfg.train_dataloader.num_batches_per_epoch}')
    # print(f'lr: {cfg.model._args_.lr}')

    # Qz: Check the validation loss for the initial pretrained model
    trainer.validate(model, datamodule=DataModule(cfg, train_dataset, val_dataset))

    trainer.fit(
        model,
        datamodule=DataModule(cfg, train_dataset, val_dataset),
    )


if __name__ == "__main__":
    main()
