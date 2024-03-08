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

from collections.abc import Callable, Iterable
from typing import Any

import numpy as np
import torch
from jaxtyping import AbstractDtype, Num


class DateTime64(AbstractDtype):
    dtypes = ["datetime64"]


class Character(AbstractDtype):
    dtypes = ["str_"]


# Data preparation
GenFunc = Callable[[], Iterable[dict[str, Any]]]
SliceableGenFunc = Callable[..., Iterable[dict[str, Any]]]


# Indexer
DateTime = DateTime64[np.ndarray, ""]
BatchedDateTime = DateTime64[np.ndarray, "batch"]
String = np.character
BatchedString = Character[np.ndarray, "batch"]
UnivarTimeSeries = Num[np.ndarray, "time"]
MultivarTimeSeries = Num[np.ndarray, "var time"]
Data = DateTime | String | UnivarTimeSeries | MultivarTimeSeries
BatchedData = (
    BatchedDateTime | BatchedString | list[UnivarTimeSeries] | list[MultivarTimeSeries]
)
FlattenedData = DateTime | String | list[UnivarTimeSeries]


# Loader
Sample = dict[str, Num[torch.Tensor, "*sample"]]
BatchedSample = dict[str, Num[torch.Tensor, "batch *sample"]]
