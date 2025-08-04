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
    """ A custom jaxtyping dtype for numpy.datetime64. """
    dtypes = ["datetime64"]


class Character(AbstractDtype):
    """ A custom jaxtyping dtype for numpy.str_. """
    dtypes = ["str_"]


# Data preparation
GenFunc = Callable[[], Iterable[dict[str, Any]]]
""" A type alias for a generator function that yields dictionaries. """
SliceableGenFunc = Callable[..., Iterable[dict[str, Any]]]
""" A type alias for a sliceable generator function that yields dictionaries. """


# Indexer
DateTime = DateTime64[np.ndarray, ""]
""" A type alias for a single numpy.datetime64 value. """
BatchedDateTime = DateTime64[np.ndarray, "batch"]
""" A type alias for a batch of numpy.datetime64 values. """
String = np.character
""" A type alias for a numpy character. """
BatchedString = Character[np.ndarray, "batch"]
""" A type alias for a batch of numpy characters. """
UnivarTimeSeries = Num[np.ndarray, "time"]
""" A type alias for a univariate time series as a numpy array. """
MultivarTimeSeries = Num[np.ndarray, "var time"]
""" A type alias for a multivariate time series as a numpy array. """
Data = DateTime | String | UnivarTimeSeries | MultivarTimeSeries
""" A type alias for any of the supported data types. """
BatchedData = (
    BatchedDateTime | BatchedString | list[UnivarTimeSeries] | list[MultivarTimeSeries]
)
""" A type alias for a batch of any of the supported data types. """
FlattenedData = DateTime | String | list[UnivarTimeSeries]
""" A type alias for flattened data, which can be a single value or a list of time series. """


# Loader
Sample = dict[str, Num[torch.Tensor, "*sample"]]
""" A type alias for a single sample, which is a dictionary of tensors. """
BatchedSample = dict[str, Num[torch.Tensor, "batch *sample"]]
""" A type alias for a batch of samples, which is a dictionary of batched tensors. """
