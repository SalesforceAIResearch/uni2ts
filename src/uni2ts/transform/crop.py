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

from collections.abc import Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np

from uni2ts.common.typing import UnivarTimeSeries

from ._base import Transformation
from ._mixin import MapFuncMixin


@dataclass
class PatchCrop(MapFuncMixin, Transformation):
    """
    Crop fields in a data_entry in the temporal dimension based on a patch_size.
    :param rng: numpy random number generator
    :param min_time_patches: minimum number of patches for time dimension
    :param max_patches: maximum number of patches for time * dim dimension (if flatten)
    :param will_flatten: whether time series fields will be flattened subsequently
    :param offset: whether to offset the start of the crop
    :param fields: fields to crop
    """

    min_time_patches: int
    max_patches: int
    will_flatten: bool = False
    offset: bool = True
    fields: tuple[str, ...] = ("target",)
    optional_fields: tuple[str, ...] = ("past_feat_dynamic_real",)

    def __post_init__(self):
        assert (
            self.min_time_patches <= self.max_patches
        ), "min_patches must be <= max_patches"
        assert len(self.fields) > 0, "fields must be non-empty"

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        a, b = self._get_boundaries(data_entry)
        self.map_func(
            partial(self._crop, a=a, b=b),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    @staticmethod
    def _crop(data_entry: dict[str, Any], field: str, a: int, b: int) -> Sequence:
        return [ts[a:b] for ts in data_entry[field]]

    def _get_boundaries(self, data_entry: dict[str, Any]) -> tuple[int, int]:
        patch_size = data_entry["patch_size"]
        field: list[UnivarTimeSeries] = data_entry[self.fields[0]]
        time = field[0].shape[0]
        nvar = (
            sum(len(data_entry[f]) for f in self.fields)
            + sum(len(data_entry[f]) for f in self.optional_fields if f in data_entry)
            if self.will_flatten
            else 1
        )

        offset = (
            np.random.randint(
                time % patch_size + 1
            )  # offset by [0, patch_size) so that the start is not always a multiple of patch_size
            if self.offset
            else 0
        )
        total_patches = (
            time - offset
        ) // patch_size  # total number of patches in time series

        # 1. max_patches should be divided by nvar if the time series is subsequently flattened
        # 2. cannot have more patches than total available patches
        max_patches = min(self.max_patches // nvar, total_patches)
        if max_patches < self.min_time_patches:
            raise ValueError(
                f"max_patches={max_patches} < min_time_patches={self.min_time_patches}"
            )

        num_patches = np.random.randint(
            self.min_time_patches, max_patches + 1
        )  # number of patches to consider
        first = np.random.randint(
            total_patches - num_patches + 1
        )  # first patch to consider

        start = offset + first * patch_size
        stop = start + num_patches * patch_size
        return start, stop


@dataclass
class EvalCrop(MapFuncMixin, Transformation):
    offset: int
    distance: int
    prediction_length: int
    context_length: int
    fields: tuple[str, ...]
    optional_fields: tuple[str, ...] = tuple()

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        a, b = self._get_boundaries(data_entry)
        self.map_func(
            partial(self._crop, a=a, b=b),  # noqa
            data_entry,
            self.fields,
            optional_fields=self.optional_fields,
        )
        return data_entry

    @staticmethod
    def _crop(data_entry: dict[str, Any], field: str, a: int, b: int) -> Sequence:
        return [ts[a : b or None] for ts in data_entry[field]]

    def _get_boundaries(self, data_entry: dict[str, Any]) -> tuple[int, int]:
        field: list[UnivarTimeSeries] = data_entry[self.fields[0]]
        time = field[0].shape[0]
        window = data_entry["window"]
        fcst_start = self.offset + window * self.distance
        a = fcst_start - self.context_length
        b = fcst_start + self.prediction_length

        if self.offset >= 0:
            assert time >= b > a >= 0
        else:
            assert 0 >= b > a >= -time

        return a, b
