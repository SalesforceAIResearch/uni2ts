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

from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

from uni2ts.common.torch_util import numpy_to_torch_dtype_dict
from uni2ts.data.loader import DataLoader, PackCollate
from uni2ts.transform import Identity, RemoveFields, Transformation, Transpose


@dataclass
class SimpleDataset(Dataset):
    size: int
    fields: tuple[str, ...] = ("target",)

    def __getitem__(self, item):
        if item >= self.size:
            raise IndexError

        return {val: np.arange(1 + item, dtype=float) for val in self.fields}

    def __len__(self):
        return self.size


@dataclass
class SimpleDataset2(Dataset):
    data: dict[str, np.ndarray]

    def __getitem__(self, item):
        if 0 > item >= len(self):
            raise IndexError
        return {k: v[item] for k, v in self.data.items()}

    def __len__(self):
        return self.data["target"].shape[0]


@dataclass
class TSDataset(Dataset):
    create_example: Callable[..., dict[str, np.ndarray]]
    size: int
    length: int
    target_dim: int = 1
    past_feat_dynamic_real_dim: Optional[int] = None
    transform: Transformation = Identity()

    def __post_init__(self):
        self.data = [
            self.create_example(
                length=self.length,
                freq="H",
                target_dim=self.target_dim,
                past_feat_dynamic_real_dim=self.past_feat_dynamic_real_dim,
            )
            for _ in range(self.size)
        ]

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        item = self.data[idx].copy()
        return self.transform(item)


@pytest.mark.parametrize("dataset_size", [10, 50, 100])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("batch_size_factor", [1, 2, 3])
@pytest.mark.parametrize("drop_last", [True, False])
@pytest.mark.parametrize("fill_last", [True, False])
@pytest.mark.timeout(1)
def test_basic(
    dataset_size: int,
    batch_size: int,
    batch_size_factor: int,
    drop_last: bool,
    fill_last: bool,
):
    packed_loader = DataLoader(
        dataset=SimpleDataset(size=dataset_size),
        batch_size=batch_size,
        batch_size_factor=batch_size_factor,
        collate_fn=PackCollate(max_length=dataset_size, seq_fields=("target",)),
        shuffle=False,
        drop_last=drop_last,
        fill_last=fill_last,
    )

    for batch in packed_loader:
        ...


@pytest.mark.parametrize("dataset_size", [10, 50, 100])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("num_epochs", [1, 2, 5])
def test_loop(dataset_size: int, batch_size: int, num_epochs: int):
    packed_loader = DataLoader(
        dataset=SimpleDataset(size=dataset_size),
        batch_size=batch_size,
        batch_size_factor=1.0,
        collate_fn=PackCollate(max_length=dataset_size, seq_fields=("target",)),
        shuffle=False,
        drop_last=False,
        fill_last=False,
    )
    for i in range(num_epochs):
        n_samples = 0
        for batch in packed_loader:
            n_samples += batch["sample_id"].max(dim=1).values.sum()
        assert n_samples == dataset_size


@pytest.mark.parametrize("dataset_size", [10, 50, 100])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("batch_size_factor", [1, 2, 3])
@pytest.mark.timeout(1)
def test_cycle_loop(
    dataset_size: int,
    batch_size: int,
    batch_size_factor: int,
):
    packed_loader = DataLoader(
        dataset=SimpleDataset(size=dataset_size),
        batch_size=batch_size,
        batch_size_factor=batch_size_factor,
        collate_fn=PackCollate(max_length=dataset_size, seq_fields=("target",)),
        shuffle=False,
        cycle=True,
    )

    n = 0
    for batch in packed_loader:
        n += batch["sample_id"].shape[0]
        if n > dataset_size * 10:
            break


@pytest.mark.parametrize("dataset_size", [10, 50, 100])
@pytest.mark.parametrize("batch_size", [1, 2, 3])
@pytest.mark.parametrize("num_batches_per_epoch", [1, 5, 10])
@pytest.mark.parametrize("num_epochs", [1, 5, 10])
def test_num_batches_per_epoch(
    dataset_size: int,
    batch_size: int,
    num_batches_per_epoch: int,
    num_epochs: int,
):
    packed_loader = DataLoader(
        dataset=SimpleDataset(size=dataset_size),
        batch_size=batch_size,
        collate_fn=PackCollate(max_length=dataset_size, seq_fields=("target",)),
        shuffle=False,
        cycle=True,
        num_batches_per_epoch=num_batches_per_epoch,
    )
    for _ in range(num_epochs):
        counter = 0
        for batch in packed_loader:
            counter += 1
        assert counter == num_batches_per_epoch


def test_multiple_fields():
    packed_loader = DataLoader(
        dataset=SimpleDataset(size=10, fields=("target", "sequence_id")),
        collate_fn=PackCollate(max_length=10, seq_fields=("target", "sequence_id")),
        batch_size=2,
        batch_size_factor=1,
        shuffle=False,
        drop_last=False,
    )

    batch = next(iter(packed_loader))

    assert all([f in batch for f in ("target", "sequence_id", "sample_id")])
    assert all(
        [batch[f].shape == (2, 10) for f in ("target", "sequence_id", "sample_id")]
    )


@pytest.mark.timeout(5)
def test_shuffle():
    while True:
        dataset = SimpleDataset(size=10, fields=("target", "sequence_id"))
        packed_loader = DataLoader(
            dataset=dataset,
            collate_fn=PackCollate(max_length=10, seq_fields=("target", "sequence_id")),
            batch_size=2,
            batch_size_factor=1,
            shuffle=True,
            drop_last=False,
        )

        batch = next(iter(packed_loader))

        first_batch_target = batch["target"][0][batch["sample_id"][0] == 1].numpy()
        first_dataset_target = dataset[0]["target"]

        if first_batch_target.shape != first_dataset_target.shape or not np.allclose(
            first_batch_target, first_dataset_target
        ):
            break

    assert True


def test_input_longer_than_max_length():
    packed_loader = DataLoader(
        dataset=SimpleDataset(size=10),
        collate_fn=PackCollate(max_length=1, seq_fields=("target",)),
        batch_size=10,
        batch_size_factor=1,
        shuffle=False,
        drop_last=False,
    )

    with pytest.raises(
        AssertionError,
        match=r"Sample length must be less than or equal to max_length \(\d+\)",
    ):
        for batch in packed_loader:
            ...


def test_non_ts_input(create_example):
    dataset = TSDataset(
        create_example=create_example,
        size=10,
        transform=RemoveFields(["item_id", "start"]) + Transpose(fields=("target",)),
        length=10,
        target_dim=1,
        past_feat_dynamic_real_dim=None,
    )
    packed_loader = DataLoader(
        dataset=dataset,
        collate_fn=PackCollate(max_length=22, seq_fields=("target", "freq")),
        batch_size=2,
        batch_size_factor=1,
        shuffle=False,
    )

    with pytest.raises(
        AssertionError,
        match=r"All fields must have the same length.",
    ):
        for batch in packed_loader:
            ...


@pytest.mark.parametrize("dtype", list(numpy_to_torch_dtype_dict.keys()))
def test_dtype(dtype: type):
    data = np.ones((1, 3), dtype=dtype)
    packed_loader = DataLoader(
        SimpleDataset2(dict(target=data)),
        batch_size=2,
        batch_size_factor=1.0,
        cycle=False,
        collate_fn=PackCollate(
            max_length=7,
            seq_fields=("target",),
            pad_func_map=dict(target=lambda *args: np.ones(*args)),
        ),
        shuffle=False,
        drop_last=False,
        fill_last=True,
    )
    iterator = iter(packed_loader)
    sample = next(iterator)

    # check PackCollate padding
    assert sample["target"][0, 3:].dtype == numpy_to_torch_dtype_dict[dtype]
    assert torch.all(sample["target"][0, 3:] == 1)
    # check _DataIterator padding
    assert sample["target"][1].dtype == numpy_to_torch_dtype_dict[dtype]
    assert torch.all(sample["target"][1] == 1)


@pytest.mark.parametrize("batch_size", [10, 15, 20])
@pytest.mark.parametrize("batch_size_factor", [0.5, 1.0, 2.0])
def test_packing_single_dim(create_example, batch_size: int, batch_size_factor: float):
    dataset = TSDataset(
        create_example=create_example,
        size=10,
        transform=(
            RemoveFields(["item_id", "freq", "start"]) + Transpose(fields=("target",))
        ),
        length=10,
        target_dim=1,
        past_feat_dynamic_real_dim=None,
    )
    packed_loader = DataLoader(
        dataset=dataset,
        collate_fn=PackCollate(max_length=22, seq_fields=("target",)),
        batch_size=batch_size,
        batch_size_factor=batch_size_factor,
        shuffle=False,
        drop_last=False,
    )

    batch = next(iter(packed_loader))

    sample_id = batch["sample_id"][0]
    target = batch["target"][0]

    first_dataset_target = dataset[0]["target"]
    second_dataset_target = dataset[1]["target"]

    assert isinstance(sample_id, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert np.all(
        sample_id.numpy()
        == np.asarray([1 for _ in range(10)] + [2 for _ in range(10)] + [0, 0])
    )
    assert np.allclose(target[sample_id == 1].numpy(), first_dataset_target)
    assert np.allclose(target[sample_id == 2].numpy(), second_dataset_target)


@pytest.mark.parametrize("target_dim", [2, 3])
@pytest.mark.parametrize("past_feat_dynamic_real_dim", [1, 2, 3])
def test_packing_multi_dim(
    create_example, target_dim: int, past_feat_dynamic_real_dim: Optional[int]
):
    dataset = TSDataset(
        create_example=create_example,
        size=10,
        transform=RemoveFields(["item_id", "freq", "start"])
        + Transpose(fields=("target",), optional_fields=("past_feat_dynamic_real",)),
        length=10,
        target_dim=target_dim,
        past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
    )
    packed_loader = DataLoader(
        dataset=dataset,
        collate_fn=PackCollate(
            max_length=22,
            seq_fields=(
                "target",
                "past_feat_dynamic_real",
            ),
        ),
        batch_size=2,
        batch_size_factor=1,
        shuffle=False,
    )

    batch = next(iter(packed_loader))

    sample_id = batch["sample_id"][0]
    target = batch["target"][0]
    past_feat_dynamic_real = batch["past_feat_dynamic_real"][0]

    first_dataset_target = dataset[0]["target"]
    second_dataset_target = dataset[1]["target"]
    first_dataset_past_feat_dynamic_real = dataset[0]["past_feat_dynamic_real"]
    second_dataset_past_feat_dynamic_real = dataset[1]["past_feat_dynamic_real"]

    assert isinstance(sample_id, torch.Tensor)
    assert isinstance(target, torch.Tensor)
    assert isinstance(past_feat_dynamic_real, torch.Tensor)
    assert np.all(
        sample_id.numpy()
        == np.asarray([1 for _ in range(10)] + [2 for _ in range(10)] + [0, 0])
    )
    assert np.allclose(target[sample_id == 1].numpy(), first_dataset_target)
    assert np.allclose(target[sample_id == 2].numpy(), second_dataset_target)
    assert np.allclose(
        past_feat_dynamic_real[sample_id == 1].numpy(),
        first_dataset_past_feat_dynamic_real,
    )
    assert np.allclose(
        past_feat_dynamic_real[sample_id == 2].numpy(),
        second_dataset_past_feat_dynamic_real,
    )
