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

import abc
from collections.abc import Iterable, Sequence

import numpy as np

from uni2ts.common.typing import BatchedData, Data


class Indexer(abc.ABC, Sequence):
    """
    Base class for all Indexers.

    An Indexer is responsible for extracting data from an underlying file format.
    """

    def __init__(self, uniform: bool = False):
        """
        :param uniform: whether the underlying data has uniform length
        """
        self.uniform = uniform

    def check_index(self, idx: int | slice | Iterable[int]):
        """
        Check the validity of a given index.

        :param idx: index to check
        :return: None
        :raises IndexError: if idx is out of bounds
        :raises NotImplementedError: if idx is not a valid type
        """
        if isinstance(idx, int):
            if idx < 0 or idx >= len(self):
                raise IndexError(f"Index {idx} out of bounds for length {len(self)}")
        elif isinstance(idx, slice):
            if idx.start is not None and idx.start < 0:
                raise IndexError(
                    f"Index {idx.start} out of bounds for length {len(self)}"
                )
            if idx.stop is not None and idx.stop >= len(self):
                raise IndexError(
                    f"Index {idx.stop} out of bounds for length {len(self)}"
                )
        elif isinstance(idx, Iterable):
            idx = np.fromiter(idx, np.int64)
            if np.logical_or(idx < 0, idx >= len(self)).any():
                raise IndexError(f"Index out of bounds for length {len(self)}")
        else:
            raise NotImplementedError(f"Unable to index on type: {type(idx)}")

    def __getitem__(
        self, idx: int | slice | Iterable[int]
    ) -> dict[str, Data | BatchedData]:
        """
        Retrive the data from the underlying storage in dictionary format.

        :param idx: index to retrieve
        :return: underlying data with given index
        """
        self.check_index(idx)

        if isinstance(idx, int):
            item = self._getitem_int(idx)
        elif isinstance(idx, slice):
            item = self._getitem_slice(idx)
        elif isinstance(idx, Iterable):
            item = self._getitem_iterable(idx)
        else:
            raise NotImplementedError(f"Unable to index on type: {type(idx)}")

        return {k: v for k, v in item.items()}

    def _getitem_slice(self, idx: slice) -> dict[str, BatchedData]:
        indices = list(range(len(self))[idx])
        return self._getitem_iterable(indices)

    @abc.abstractmethod
    def _getitem_int(self, idx: int) -> dict[str, Data]: ...

    @abc.abstractmethod
    def _getitem_iterable(self, idx: Iterable[int]) -> dict[str, BatchedData]: ...

    def get_uniform_probabilities(self) -> np.ndarray:
        """
        Obtains uniform probability distribution over all time series.

        :return: uniform probability distribution
        """
        return np.ones(len(self)) / len(self)

    def get_proportional_probabilities(self, field: str = "target") -> np.ndarray:
        """
        Obtain proportion of each time series based on number of time steps.

        :param field: field name to measure time series length
        :return: proportional probabilities
        """
        if self.uniform:
            return self.get_uniform_probabilities()

        lengths = np.asarray([sample[field].shape[-1] for sample in self])
        probs = lengths / lengths.sum()
        return probs
