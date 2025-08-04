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
"""
A type alias for a sampler function. A sampler function takes an integer or a numpy array `n`
and returns a random integer or array of integers between 1 and `n` (inclusive).
"""


def uniform_sampler(n: int | np.ndarray) -> int | np.ndarray:
    """
    Samples a random integer or array of integers uniformly from [1, n].

    Args:
        n (int | np.ndarray): The upper bound for the sampling (inclusive).

    Returns:
        int | np.ndarray: A random integer or array of integers.
    """
    return np.random.randint(1, n + 1)


def binomial_sampler(n: int | np.ndarray, p: float = 0.5) -> int | np.ndarray:
    """
    Samples a random integer or array of integers from a binomial distribution.
    The sample is drawn from Bin(n-1, p) and then shifted by 1 to be in the range [1, n].

    Args:
        n (int | np.ndarray): The number of trials (upper bound).
        p (float, optional): The probability of success. Defaults to 0.5.

    Returns:
        int | np.ndarray: A random integer or array of integers.
    """
    return np.random.binomial(n - 1, p) + 1


def beta_binomial_sampler(
    n: int | np.ndarray, a: float = 1, b: float = 1
) -> int | np.ndarray:
    """
    Samples a random integer or array of integers from a beta-binomial distribution.
    This is a binomial distribution where the probability of success `p` is drawn from a
    beta distribution with parameters `a` and `b`.

    Note:
        This is equivalent to the uniform sampler when a = b = 1.

    Args:
        n (int | np.ndarray): The number of trials (upper bound).
        a (float, optional): The alpha parameter of the beta distribution. Defaults to 1.
        b (float, optional): The beta parameter of the beta distribution. Defaults to 1.

    Returns:
        int | np.ndarray: A random integer or array of integers.
    """
    # equivalent to uniform_sampler when a = b = 1
    if isinstance(n, np.ndarray):
        p = np.random.beta(a, b, size=n.shape)
    else:
        p = np.random.beta(a, b)
    return np.random.binomial(n - 1, p) + 1


def get_sampler(distribution: str, **kwargs) -> Sampler:
    """
    A factory function that returns a sampler function based on a string identifier.

    Args:
        distribution (str): The name of the distribution to use for sampling.
                            Can be "uniform", "binomial", or "beta_binomial".
        **kwargs: Additional keyword arguments to pass to the sampler function.
                  For "binomial", this can be `p`.
                  For "beta_binomial", this can be `a` and `b`.

    Returns:
        Sampler: A sampler function.

    Raises:
        NotImplementedError: If the specified distribution is not implemented.
    """
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
