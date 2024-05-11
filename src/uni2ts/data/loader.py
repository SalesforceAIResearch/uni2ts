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

import itertools
from collections import defaultdict, deque
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import NamedTuple, Optional

import numpy as np
import torch
from jaxtyping import Bool, Int
from torch.utils.data import DataLoader as TorchDataLoader
from torch.utils.data import Dataset, Sampler, default_collate, default_convert

from uni2ts.common.typing import BatchedSample, Sample


@dataclass
class Collate:
    max_length: Optional[int]
    seq_fields: tuple[str, ...]
    pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]] = field(
        default_factory=dict
    )
    target_field: str = "target"

    def __post_init__(self):
        self.pad_func_map = defaultdict(self._default_pad_func) | self.pad_func_map

    @staticmethod
    def _default_pad_func() -> Callable[[Sequence[int], np.dtype], np.ndarray]:
        return np.zeros

    def __call__(self, batch: list[Sample]) -> BatchedSample:
        raise NotImplementedError


class PadCollate(Collate):
    def __call__(self, batch: list[Sample]) -> BatchedSample:
        assert all(
            [
                len(sample[self.target_field]) == len(sample[key])
                for sample in batch
                for key in self.seq_fields
            ]
        ), "All fields must have the same length."
        assert all(
            [len(sample[self.target_field]) <= self.max_length for sample in batch]
        ), f"Sample length must be less than or equal to max_length ({self.max_length})"

        sample_id = self.get_sample_id(batch)
        padded_batch = self.pad_samples(batch)
        merged_batch = padded_batch | dict(sample_id=sample_id)
        return merged_batch

    def pad_samples(self, batch: list[Sample]) -> BatchedSample:
        for sample in batch:
            length = len(sample[self.target_field])
            for key in self.seq_fields:
                sample[key] = torch.cat(
                    [
                        default_convert(sample[key]),
                        default_convert(
                            self.pad_func_map[key](
                                (self.max_length - length,) + sample[key].shape[1:],
                                sample[key].dtype,
                            )
                        ),
                    ]
                )
        return default_collate(batch)

    def get_sample_id(self, batch: list[Sample]) -> Int[torch.Tensor, "batch seq"]:
        sample_id = torch.stack(
            [
                torch.cat([torch.ones(length), torch.zeros(self.max_length - length)])
                for sample in batch
                if (length := len(sample[self.target_field]))
            ]
        ).to(torch.long)
        return sample_id


class PackCollate(Collate):
    def __call__(self, batch: list[Sample]) -> BatchedSample:
        assert all(
            [
                len(sample[self.target_field]) == len(sample[key])
                for sample in batch
                for key in self.seq_fields
            ]
        ), "All fields must have the same length."
        assert all(
            [len(sample[self.target_field]) <= self.max_length for sample in batch]
        ), f"Sample length must be less than or equal to max_length ({self.max_length})"

        # Add field 'sample_id', indicating id for the samples in each packed group/bin (1 item in forward's batch)
        # For each bin, its sample_id starts from 1, ends at the max number samples in it.
        # For patches padded in the end, their sample_id is 0.
        # Also pad the other fields with 0.
        packed_batch, bin_spaces = self.first_fit_decreasing_bin_packing(batch)
        sample_id = self.get_sample_id(packed_batch, bin_spaces)
        merged_batch = self.merge_batch(packed_batch, bin_spaces) | dict(
            sample_id=sample_id
        )
        return merged_batch

    def first_fit_decreasing_bin_packing(
        self,
        batch: list[Sample],  # list of dict
    ) -> tuple[list[list[Sample]], Int[np.ndarray, "batch"]]:
        # Sort the batch by decreasing sequence length
        batch = sorted(
            batch, key=lambda sample: len(sample[self.target_field]), reverse=True
        )
        bin_spaces: Int[np.ndarray, "batch"] = np.full(len(batch), self.max_length)  # 2 args are shape and fill_value
        packed_batch: list[list[Sample]] = [[]]

        for sample in batch:
            length = len(sample[self.target_field])
            criterion: Bool[np.ndarray, "batch"] = bin_spaces - length >= 0
            bin_id: int = criterion.argmax()
            if len(packed_batch) <= bin_id:
                if len(packed_batch) != bin_id:
                    raise ValueError
                packed_batch.append([])

            packed_batch[bin_id].append(sample)
            bin_spaces[bin_id] -= length

        return packed_batch, bin_spaces[: len(packed_batch)]

    def get_sample_id(
        self, batch: list[list[Sample]], bin_spaces: Int[np.ndarray, "batch"]
    ) -> Int[torch.Tensor, "batch seq"]:
        sample_id = torch.stack(
            [
                torch.cat(
                    [
                        torch.ones(len(sample[self.target_field])) * (idx + 1)
                        for idx, sample in enumerate(bin_)
                    ]
                    + [torch.zeros(space)],  # padding
                )
                for bin_, space in zip(batch, bin_spaces)
            ]
        ).to(torch.long)
        return sample_id

    def merge_batch(
        self, batch: list[list[Sample]], bin_spaces: Int[np.ndarray, "batch"]
    ) -> BatchedSample:
        batch = {
            key: torch.stack(
                [
                    torch.cat(
                        [default_convert(sample[key]) for sample in bin_]
                        + [
                            default_convert(
                                self.pad_func_map[key](
                                    (space,) + bin_[0][key].shape[1:],
                                    bin_[0][key].dtype,
                                )
                            )
                        ]
                    )
                    for bin_, space in zip(batch, bin_spaces)
                ],
            )
            for key in self.seq_fields
        }
        return batch


@dataclass
class SliceableBatchedSample:
    data: BatchedSample

    def __post_init__(self):
        assert all(
            [
                len(self.data[key]) == len(self.data[next(iter(self.data))])
                for key in self.data.keys()
            ]
        )

    def __len__(self) -> int:
        return len(self.data[next(iter(self.data))])

    def __getitem__(self, item: slice) -> "SliceableBatchedSample":
        return SliceableBatchedSample(
            {key: self.data[key][item] for key in self.data.keys()}
        )


class Metadata(NamedTuple):
    shape: tuple[int, ...]
    dtype: torch.dtype


@dataclass
class BatchedSampleQueue:
    container: deque[SliceableBatchedSample] = field(default_factory=deque)
    schema: Optional[dict[str, Metadata]] = None

    def _check_schema(self, batch: SliceableBatchedSample):
        if self.schema is None:
            self.schema = {
                key: Metadata(
                    shape=tuple(batch.data[key].shape[1:]), dtype=batch.data[key].dtype
                )
                for key in batch.data.keys()
            }
        else:
            assert all(
                [
                    (key in batch.data)
                    and (metadata.shape == tuple(batch.data[key].shape[1:]))
                    and (metadata.dtype == batch.data[key].dtype)
                    for key, metadata in self.schema.items()
                ]
            ), "batch must have the same schema as the first batch"

    def append(self, batch: SliceableBatchedSample | BatchedSample):
        if not isinstance(batch, SliceableBatchedSample):
            batch = SliceableBatchedSample(batch)
        self._check_schema(batch)
        self.container.append(batch)

    def appendleft(self, batch: SliceableBatchedSample | BatchedSample):
        if not isinstance(batch, SliceableBatchedSample):
            batch = SliceableBatchedSample(batch)
        self._check_schema(batch)
        self.container.appendleft(batch)

    def popleft(self, size: int) -> BatchedSample:
        if size > len(self):
            raise ValueError(
                f"pop size ({size}) must be less than or equal to queue size ({len(self)})"
            )

        out = BatchedSampleQueue()
        while len(out) < size:
            curr = self.container.popleft()
            if len(out) + len(curr) > size:
                self.appendleft(curr[size - len(out) :])
                curr = curr[: size - len(out)]
            out.append(curr)
        return out.as_batched_data()

    def as_batched_data(self) -> BatchedSample:
        return {
            key: torch.cat([batch.data[key] for batch in self.container], dim=0)
            for key in self.schema.keys()
        }

    def __len__(self) -> int:
        return sum(len(batch) for batch in self.container)


@dataclass
class _BatchedSampleIterator:
    dataloader_iter: Iterator[BatchedSample]
    batch_size: int
    drop_last: bool
    fill_last: bool
    pad_func_map: Optional[dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]]]=None

    def __post_init__(self):
        self.queue = BatchedSampleQueue()

    def __iter__(self):
        return self

    def __next__(self) -> BatchedSample:
        while (data := self._next_batch()) is None:
            continue
        return data

    def _next_batch(self) -> Optional[BatchedSample]:
        if len(self.queue) < self.batch_size:
            try:
                data = next(self.dataloader_iter)
                self.queue.append(data)
                return None
            except StopIteration:
                if self.drop_last or len(self.queue) == 0:
                    raise StopIteration
                elif self.fill_last:
                    self._pad_queue(self.batch_size - len(self.queue))

        batch = self.queue.popleft(min(self.batch_size, len(self.queue)))
        return batch

    def _pad_queue(self, size: int):
        if self.queue.schema is None:
            raise ValueError("schema must be set before padding")
        padding = {
            key: default_convert(
                self.pad_func_map[key]((size,) + metadata.shape, np.dtype(np.float32))
            ).to(metadata.dtype)
            for key, metadata in self.queue.schema.items()
        }
        self.queue.append(padding)

    def has_next(self) -> bool:
        if len(self.queue) < self.batch_size:
            try:
                next_batch = next(self)
                self.queue.appendleft(next_batch)
            except StopIteration:
                return False
        return True

pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]] = {
    "target": np.zeros,
    "observed_mask": np.zeros,
    "time_id": np.zeros,
    "variate_id": np.zeros,
    "prediction_mask": np.zeros,
    "patch_size": np.zeros,
}

class DataLoader:  # Used in Pretrain and Finetune.
    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        batch_size_factor: float = 1.0,
        cycle: bool = False,
        num_batches_per_epoch: Optional[int] = None,
        shuffle: bool = False,
        sampler: Optional[Sampler] = None,
        num_workers: int = 0,
        collate_fn: Optional[Collate] = None,
        pin_memory: bool = False,
        drop_last: bool = True,
        fill_last: bool = False,
        worker_init_fn: Optional[Callable[[int], None]] = None,
        prefetch_factor: int = 2,
        persistent_workers: bool = False,
    ):
        if num_batches_per_epoch is not None:
            assert cycle, "can only set 'num_batches_per_epoch' when 'cycle=True'"

        self.dataloader = TorchDataLoader(
            dataset=dataset,
            batch_size=int(batch_size * batch_size_factor),  # number of samples (unpacked) for a mini batch
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            drop_last=False,
            worker_init_fn=worker_init_fn,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers and num_workers > 0,
        )
        self.batch_size = batch_size  # number of packed samples/bins in a mini batch (bs in forward)
        self.cycle = cycle
        self.num_batches_per_epoch = num_batches_per_epoch
        self.collate_fn = collate_fn
        self.drop_last = drop_last
        self.fill_last = fill_last
        self.iterator: Optional[_BatchedSampleIterator] = None

    def __iter__(self) -> Iterator:
        if self.iterator is None or not self.iterator.has_next():
            dataloader_iter = (
                iter(self.dataloader)
                if not self.cycle
                else itertools.chain.from_iterable(itertools.repeat(self.dataloader))
            )
            self.iterator = _BatchedSampleIterator(
                dataloader_iter=dataloader_iter,
                batch_size=self.batch_size,
                drop_last=self.drop_last,
                fill_last=self.fill_last,
                pad_func_map=pad_func_map if self.collate_fn is None else self.collate_fn.pad_func_map,  # self.collate_fn.pad_func_map
            )
        return itertools.islice(self.iterator, self.num_batches_per_epoch)

    @property
    def worker_init_fn(self) -> Optional[Callable[[int], None]]:
        return self.dataloader.worker_init_fn

    @worker_init_fn.setter
    def worker_init_fn(self, worker_init_fn: Optional[Callable[[int], None]]):
        self.dataloader.worker_init_fn = worker_init_fn
