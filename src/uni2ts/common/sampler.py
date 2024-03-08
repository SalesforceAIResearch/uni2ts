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

from collections.abc import Callable
from functools import partial
from typing import cast

import numpy as np

Sampler = Callable[[int | np.ndarray], int | np.ndarray]


def uniform_sampler(n: int | np.ndarray) -> int | np.ndarray:
    return np.random.randint(1, n + 1)


def binomial_sampler(n: int | np.ndarray, p: float = 0.5) -> int | np.ndarray:
    return np.random.binomial(n - 1, p) + 1


def beta_binomial_sampler(
    n: int | np.ndarray, a: float = 1, b: float = 1
) -> int | np.ndarray:
    # equivalent to uniform_sampler when a = b = 1
    if isinstance(n, np.ndarray):
        p = np.random.beta(a, b, size=n.shape)
    else:
        p = np.random.beta(a, b)
    return np.random.binomial(n - 1, p) + 1


def get_sampler(distribution: str, **kwargs) -> Sampler:
    if distribution == "uniform":
        return uniform_sampler
    elif distribution == "binomial":
        p = kwargs.get("p", 0.5)
        return cast(Sampler, partial(binomial_sampler, p=p))
    elif distribution == "beta_binomial":
        a = kwargs.get("a", 1)
        b = kwargs.get("b", 1)
        return cast(Sampler, partial(beta_binomial_sampler, a=a, b=b))
    else:
        raise NotImplementedError(f"distribution {distribution} not implemented")
