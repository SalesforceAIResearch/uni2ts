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
    """
    An abstract base class for collate functions, designed to be used with PyTorch's DataLoader.
    It defines a common interface for padding and packing sequences.

    Args:
        max_length (Optional[int]): The maximum length to pad or pack sequences to.
        seq_fields (tuple[str, ...]): A tuple of field names that contain sequence data.
        pad_func_map (dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]], optional):
            A dictionary mapping field names to padding functions. Defaults to a dictionary
            that uses `np.zeros` for all fields.
        target_field (str, optional): The name of the target field, used to determine the
            length of sequences. Defaults to "target".
    """

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
    """
    A collate function that pads uneven sequences to a `max_length`. The padding
    is performed using the functions defined in `pad_func_map`.

    This collate function also creates a `sample_id` tensor that can be used to
    distinguish between real data and padding.
    """

    def __call__(self, batch: list[Sample]) -> BatchedSample:
        """
        Pads a batch of samples and returns a batched sample.
        """
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
        """
        Pads each sample in the batch to `max_length`.
        """
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
    """
    A collate function that packs uneven sequences into a smaller number of sequences
    of `max_length` using the first-fit-decreasing bin packing strategy. This can
    be more efficient than padding when dealing with sequences of highly variable
    lengths.
    """

    def __call__(self, batch: list[Sample]) -> BatchedSample:
        """
        Packs a batch of samples and returns a batched sample.
        """
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

        packed_batch, bin_spaces = self.first_fit_decreasing_bin_packing(batch)
        sample_id = self.get_sample_id(packed_batch, bin_spaces)
        merged_batch = self.merge_batch(packed_batch, bin_spaces) | dict(
            sample_id=sample_id
        )
        return merged_batch

    def first_fit_decreasing_bin_packing(
        self,
        batch: list[Sample],
    ) -> tuple[list[list[Sample]], Int[np.ndarray, "batch"]]:
        """
        Implements the first-fit-decreasing bin packing algorithm.
        1. Sort the batch by sequence length in descending order.
        2. Initialize a list of bins, each with a capacity of `max_length`.
        3. Iterate through the sorted batch and place each sample into the first
           bin that has enough remaining space.

        Args:
            batch (list[Sample]): A list of samples to pack.

        Returns:
            tuple[list[list[Sample]], Int[np.ndarray, "..."]]: A tuple containing the
                packed batch and the remaining space in each bin.
        """
        batch = sorted(
            batch, key=lambda sample: len(sample[self.target_field]), reverse=True
        )
        bin_spaces: Int[np.ndarray, "batch"] = np.full(len(batch), self.max_length)
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
        """
        Create an array of integers representing the sample id in a sequence.
        Sample id starts from 1, and 0 represents padding.

        :param batch: packed samples
        :param bin_spaces: length of each bin
        :return: integer array, indicating the sample index given a sequence
        """
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
        """
        Merges the packed samples into a single batched sample.
        """
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
    """
    A wrapper around a `BatchedSample` that allows it to be sliced.

    Args:
        data (BatchedSample): The batched sample to wrap.
    """

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
    """
    A queue data structure for storing and managing batched samples. It ensures that
    all samples in the queue have the same schema (i.e., the same keys, shapes,
    and dtypes).

    Args:
        container (deque[SliceableBatchedSample], optional): The internal deque to
            store samples. Defaults to an empty deque.
        schema (Optional[dict[str, Metadata]], optional): The schema for the samples
            in the queue. If not provided, it is inferred from the first sample added.
            Defaults to None.
    """

    container: deque[SliceableBatchedSample] = field(default_factory=deque)
    schema: Optional[dict[str, Metadata]] = None

    def _check_schema(self, batch: SliceableBatchedSample):
        """
        Checks if a batch conforms to the queue's schema. If the schema is not yet
        defined, it is inferred from the batch.
        """
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
        """
        Appends a batch to the end of the queue.
        """
        if not isinstance(batch, SliceableBatchedSample):
            batch = SliceableBatchedSample(batch)
        self._check_schema(batch)
        self.container.append(batch)

    def appendleft(self, batch: SliceableBatchedSample | BatchedSample):
        """
        Appends a batch to the start of the queue.
        """
        if not isinstance(batch, SliceableBatchedSample):
            batch = SliceableBatchedSample(batch)
        self._check_schema(batch)
        self.container.appendleft(batch)

    def popleft(self, size: int) -> BatchedSample:
        """
        Pops a batch of a given size from the start of the queue.
        """
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
        """
        Returns the contents of the queue as a single `BatchedSample`.
        """
        return {
            key: torch.cat([batch.data[key] for batch in self.container], dim=0)
            for key in self.schema.keys()
        }

    def __len__(self) -> int:
        """
        Returns the total number of samples in the queue.
        """
        return sum(len(batch) for batch in self.container)


@dataclass
class _BatchedSampleIterator:
    """
    An iterator that returns batched samples with a fixed batch size. It wraps another
    dataloader iterator and uses a queue to buffer samples, ensuring that each
    batch returned has the specified `batch_size`.

    Args:
        dataloader_iter (Iterator[BatchedSample]): The underlying dataloader iterator.
        batch_size (int): The desired batch size.
        drop_last (bool): Whether to drop the last batch if it is smaller than `batch_size`.
        fill_last (bool): Whether to fill the last batch with padding if it is smaller
            than `batch_size`.
        pad_func_map (dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]]):
            A dictionary mapping field names to padding functions.
    """

    dataloader_iter: Iterator[BatchedSample]
    batch_size: int
    drop_last: bool
    fill_last: bool
    pad_func_map: dict[str, Callable[[Sequence[int], np.dtype], np.ndarray]]

    def __post_init__(self):
        self.queue = BatchedSampleQueue()

    def __iter__(self):
        return self

    def __next__(self) -> BatchedSample:
        while (data := self._next_batch()) is None:
            continue
        return data

    def _next_batch(self) -> Optional[BatchedSample]:
        """
        Returns the next batch of samples, or None if more data is needed from the
        underlying iterator.
        """
        if len(self.queue) < self.batch_size:
            # check if there are sufficient samples in the queue
            # if not, extract the next batch from dataloader_iter and return None
            try:
                data = next(self.dataloader_iter)
                self.queue.append(data)
                return None
            except StopIteration:
                # no more batches from the dataloader_iter
                # check for drop_last and fill_last strategy
                if self.drop_last or len(self.queue) == 0:
                    raise StopIteration
                elif self.fill_last:
                    self._pad_queue(self.batch_size - len(self.queue))

        batch = self.queue.popleft(min(self.batch_size, len(self.queue)))
        return batch

    def _pad_queue(self, size: int):
        """
        Pads the queue with a given number of padding samples.
        """
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
        """
        Checks if the iterator has more batches.
        """
        if len(self.queue) < self.batch_size:
            try:
                next_batch = next(self)
                self.queue.appendleft(next_batch)
            except StopIteration:
                return False
        return True


class DataLoader:
    """
    A wrapper around PyTorch's DataLoader that adds support for packing, cycling,
    and a fixed number of batches per epoch.

    Args:
        dataset (Dataset): The dataset to load from.
        batch_size (int): The number of samples per batch.
        batch_size_factor (float, optional): A factor to multiply the `batch_size` by
            when creating the underlying PyTorch DataLoader. This is useful when
            using packing, as the underlying dataloader can have a larger batch size.
            Defaults to 1.0.
        cycle (bool, optional): Whether to cycle the dataloader infinitely. Defaults to False.
        num_batches_per_epoch (Optional[int], optional): The number of batches per epoch.
            If specified, the dataloader will be cycled. Defaults to None.
        shuffle (bool, optional): Whether to shuffle the data. Defaults to False.
        sampler (Optional[Sampler], optional): A sampler to use. Defaults to None.
        num_workers (int, optional): The number of worker processes to use. Defaults to 0.
        collate_fn (Optional[Collate], optional): A collate function to use. Defaults to None.
        pin_memory (bool, optional): Whether to pin memory. Defaults to False.
        drop_last (bool, optional): Whether to drop the last batch. Defaults to True.
        fill_last (bool, optional): Whether to fill the last batch with padding. Defaults to False.
        worker_init_fn (Optional[Callable[[int], None]], optional): A worker init function.
            Defaults to None.
        prefetch_factor (int, optional): The prefetch factor. Defaults to 2.
        persistent_workers (bool, optional): Whether to use persistent workers. Defaults to False.
    """

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
        """
        Only wrapper specific arguments are documented below.
        See https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        for documentation on PyTorch DataLoader arguments.

        :param batch_size_factor: multiply the batch_size given to PyTorch's DataLoader.
        :param cycle: whether to cycle dataloader infinitely.
        :param num_batches_per_epoch: number of batches per epoch.
        :param fill_last: whether to fill the last batch with padding.
        """
        if num_batches_per_epoch is not None:
            assert cycle, "can only set 'num_batches_per_epoch' when 'cycle=True'"

        self.dataloader = TorchDataLoader(
            dataset=dataset,
            batch_size=int(batch_size * batch_size_factor),
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
        self.batch_size = batch_size
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
                pad_func_map=self.collate_fn.pad_func_map,
            )
        return itertools.islice(self.iterator, self.num_batches_per_epoch)

    @property
    def worker_init_fn(self) -> Optional[Callable[[int], None]]:
        """
        A getter for the `worker_init_fn` of the underlying dataloader.
        """
        return self.dataloader.worker_init_fn

    @worker_init_fn.setter
    def worker_init_fn(self, worker_init_fn: Optional[Callable[[int], None]]):
        """
        A setter for the `worker_init_fn` of the underlying dataloader.
        """
        self.dataloader.worker_init_fn = worker_init_fn
