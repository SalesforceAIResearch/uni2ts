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

from functools import partial

import numpy as np
import pytest
from scipy import stats

from uni2ts.common.sampler import Sampler, get_sampler


@pytest.mark.parametrize("distribution", ["uniform", "binomial", "beta_binomial"])
@pytest.mark.parametrize("val", [1, 5, 10, 15, 20])
def test_samplers_support(distribution: str, val: int):
    sampler = get_sampler(distribution)
    np.random.seed(0)
    n = sampler(np.ones(val * 1000, dtype=int) * val)
    assert np.logical_and(1 <= n, n <= val).all(), (
        f"Sampler {distribution} should only support values in range [1, {val}]."
        f"Got min val of {n.min()} and max val of {n.max()} instead."
    )


def check_pmf(
    sampler: Sampler, pmf_fn: partial[np.ndarray], val: int, num_samples: int = 2000000
):
    support = np.arange(val)

    # get sampler's empirical pmf
    np.random.seed(0)

    samples = sampler(np.ones(num_samples, dtype=int) * val) - 1
    counts = np.expand_dims(samples, axis=1) == np.expand_dims(support, axis=0)
    counts = counts.sum(axis=0)

    # get true pmf
    pmf = pmf_fn(support)

    assert np.all(~np.isnan(pmf)), "pmf should not contain NaNs"

    res = stats.chisquare(
        f_obs=counts, f_exp=pmf * num_samples
    )  # The chi-square test null hypothesis: Categorical data has the given frequencies
    assert res.pvalue > 0.05, (
        "Null hypothesis of Chi-squared test indicates that the observed empirical pmf "
        "is the same as the true pmf. P-value should be greater than 0.05 "
        f"to avoid rejecting the null hypothesis. Got p-value of {res.pvalue} instead."
    )


@pytest.mark.parametrize(
    "val",
    [
        5,
        pytest.param(10, marks=pytest.mark.slow),
        pytest.param(15, marks=pytest.mark.slow),
        pytest.param(20, marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize("p", [i / 10 for i in range(1, 10, 2)])
def test_uniform_pmf(val: int, p: float):
    sampler = get_sampler("uniform")
    check_pmf(sampler, partial(stats.randint.pmf, low=0, high=val), val)


@pytest.mark.parametrize(
    "val",
    [
        5,
        pytest.param(10, marks=pytest.mark.slow),
        pytest.param(15, marks=pytest.mark.slow),
        pytest.param(20, marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize("p", [i / 10 for i in range(1, 10, 2)])
def test_binomial_pmf(val: int, p: float):
    sampler = get_sampler("binomial", p=p)
    check_pmf(sampler, partial(stats.binom.pmf, n=val - 1, p=p), val)


@pytest.mark.parametrize(
    "val",
    [
        5,
        pytest.param(10, marks=pytest.mark.slow),
        pytest.param(15, marks=pytest.mark.slow),
        pytest.param(20, marks=pytest.mark.slow),
    ],
)
@pytest.mark.parametrize("a", [0.1, 1, 10])
@pytest.mark.parametrize("b", [0.1, 1, 10])
def test_beta_binomial_pmf(val: int, a: float, b: float):
    sampler = get_sampler("beta_binomial", a=a, b=b)
    check_pmf(sampler, partial(stats.betabinom.pmf, n=val - 1, a=a, b=b), val)
