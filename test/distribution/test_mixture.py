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

import numpy as np
import pytest
import torch
from einops import repeat
from torch.distributions import (
    Categorical,
    Distribution,
    Gamma,
    NegativeBinomial,
    Normal,
    Poisson,
    StudentT,
)
from torch.distributions.utils import logits_to_probs, probs_to_logits

from uni2ts.distribution import NormalOutput, StudentTOutput
from uni2ts.distribution.mixture import Mixture, MixtureOutput

BINS = np.linspace(-5, 5, 100)
NUM_SAMPLES = 1_000
NUM_SAMPLES_LARGE = 1_000_000


def histogram(samples: np.ndarray) -> np.ndarray:
    h, _ = np.histogram(samples, bins=BINS, density=True)
    return h


def diff(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(x - y))


def get_normal_distribution(
    shape: tuple[int, ...],
    loc: float = 0.0,
    scale: float = 1.0,
) -> Normal:
    loc = torch.ones(shape) * loc
    scale = torch.ones(shape) * scale
    return Normal(loc=loc, scale=scale)


def get_student_t_distribution(
    shape: tuple[int, ...],
    loc: float = 0.0,
    scale: float = 1.0,
    df: float = 2.1,
) -> StudentT:
    loc = torch.ones(shape) * loc
    scale = torch.ones(shape) * scale
    df = torch.ones(shape) * df
    return StudentT(loc=loc, scale=scale, df=df)


@pytest.mark.parametrize(
    "distr1_func, distr2_func, logits",
    [
        (get_normal_distribution, get_normal_distribution, (1.0, 1.0)),
        (get_normal_distribution, get_normal_distribution, (1.0, 2.0)),
        (
            partial(get_student_t_distribution, scale=1e-1),
            partial(get_student_t_distribution, scale=1e-2),
            (1.0, 1.0),
        ),
        (
            partial(get_student_t_distribution, scale=1e-1),
            partial(get_student_t_distribution, scale=1e-2),
            (1.0, 2.0),
        ),
        (
            get_normal_distribution,
            partial(get_student_t_distribution, scale=1e-1),
            (1.0, 1.0),
        ),
    ],
)
@pytest.mark.parametrize("batch_shape", [tuple(), (1,), (1, 2)])
def test_mixture_sample_stats(
    distr1_func: Callable[[tuple[int, ...]], Distribution],
    distr2_func: Callable[[tuple[int, ...]], Distribution],
    logits: tuple[float, ...],
    batch_shape: tuple[int, ...],
):
    logits = repeat(
        torch.as_tensor(logits), f"c -> {' '.join(map(str, batch_shape))} c"
    )

    distr1 = distr1_func(batch_shape)
    distr2 = distr2_func(batch_shape)
    samples1 = distr1.sample(torch.Size([NUM_SAMPLES_LARGE]))
    samples2 = distr2.sample(torch.Size([NUM_SAMPLES_LARGE]))
    rand = torch.rand(size=(NUM_SAMPLES_LARGE, *batch_shape))
    choice = rand < logits_to_probs(logits)[..., 0]
    samples_ref = torch.where(choice, samples1, samples2)
    mixture = Mixture(
        weights=Categorical(logits=logits),
        components=[distr1, distr2],
        validate_args=True,
    )
    samples_mix = mixture.sample(torch.Size([NUM_SAMPLES_LARGE]))
    assert samples1.shape == samples2.shape == samples_mix.shape == samples_ref.shape
    assert diff(histogram(samples_mix.numpy()), histogram(samples_ref.numpy())) < 0.05
    assert torch.allclose(mixture.mean, samples_mix.mean(dim=0), atol=1e-1)
    assert torch.allclose(mixture.variance, samples_mix.var(dim=0), atol=2e-1)


@pytest.mark.parametrize(
    "distribution, values_outside_support",
    [
        (
            NegativeBinomial(
                total_count=torch.as_tensor(1.0),
                logits=torch.as_tensor(1.0),
                validate_args=True,
            ),
            torch.as_tensor([1.1]),
        ),
        (
            NegativeBinomial(
                total_count=torch.as_tensor(1.0),
                logits=torch.as_tensor(1.0),
                validate_args=True,
            ),
            torch.as_tensor([-0.1]),
        ),
        (
            Gamma(
                concentration=torch.as_tensor(0.9),
                rate=torch.as_tensor(2.0),
                validate_args=True,
            ),
            torch.as_tensor([-1.0]),
        ),
        (
            Poisson(rate=torch.as_tensor(1.0), validate_args=True),
            torch.as_tensor([-1.0]),
        ),
        (
            Poisson(rate=torch.as_tensor(1.0), validate_args=True),
            torch.as_tensor([0.1]),
        ),
    ],
)
@pytest.mark.parametrize("p", [p.item() for p in np.linspace(0.1, 0.9, 9)])
def test_mixture_log_prob(
    distribution: Distribution,
    values_outside_support: torch.Tensor,
    p: float,
):
    normal = get_normal_distribution(distribution.batch_shape)
    mixture = Mixture(
        weights=Categorical(logits=probs_to_logits(torch.tensor([p, 1 - p]))),
        components=[normal, distribution],
        validate_args=True,
    )
    lp = mixture.log_prob(values_outside_support)
    assert torch.allclose(
        lp,
        torch.log(torch.as_tensor(p)) + normal.log_prob(values_outside_support),
        atol=1e-6,
    )


def test_mixture_grads():
    logits = torch.tensor([[1.0, 1.0, 1.0, 1.0]], requires_grad=True)
    loc = torch.tensor([0.0], requires_grad=True)
    scale = torch.tensor([1.0], requires_grad=True)
    df = torch.tensor([2.1], requires_grad=True)
    total_count = torch.tensor([1.0], requires_grad=True)
    probs = torch.tensor([0.5], requires_grad=True)
    concentration = torch.tensor([0.9], requires_grad=True)
    gamma_rate = torch.tensor([2.0], requires_grad=True)
    poisson_rate = torch.tensor([1.0], requires_grad=True)

    mixture = Mixture(
        weights=Categorical(logits=logits),
        components=[
            StudentT(df=df, loc=loc, scale=scale),
            NegativeBinomial(total_count, probs),
            Gamma(concentration=concentration, rate=gamma_rate),
            Poisson(rate=poisson_rate),
        ],
        validate_args=True,
    )

    log_prob = mixture.log_prob(torch.tensor([-1.0, 1.0, 2.0]))
    log_prob.sum().backward()
    assert not torch.isnan(logits.grad).any()
    assert not torch.isnan(loc.grad).any()
    assert not torch.isnan(scale.grad).any()
    assert not torch.isnan(df.grad).any()
    assert not torch.isnan(total_count.grad).any()
    assert not torch.isnan(probs.grad).any()
    assert not torch.isnan(concentration.grad).any()
    assert not torch.isnan(gamma_rate.grad).any()
    assert not torch.isnan(poisson_rate.grad).any()


@pytest.mark.parametrize("batch_shape", [tuple(), (1,), (1, 2)])
@pytest.mark.parametrize("hidden_dim", [32, 64])
@pytest.mark.parametrize("patch_sizes", [(1,), (1, 2)])
@pytest.mark.parametrize("sample_shape", [tuple(), (1,), (1, 2)])
def test_mixture_output(
    batch_shape: tuple[int, ...],
    hidden_dim: int,
    patch_sizes: tuple[int, ...],
    sample_shape: tuple[int, ...],
):
    distr_output = MixtureOutput(
        components=[
            StudentTOutput(),
            NormalOutput(),
        ]
    )
    proj = distr_output.get_param_proj(hidden_dim, out_features=patch_sizes)
    distr_kwargs = proj(
        torch.randn(batch_shape + (hidden_dim,)),
        torch.as_tensor(np.random.choice(patch_sizes, batch_shape)),
    )
    distr = distr_output.distribution(distr_kwargs)
    sample = distr.sample(torch.Size(sample_shape))
    assert sample.shape == (*sample_shape, *batch_shape, max(patch_sizes))
