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

import pytest
import torch
from torch.distributions import NegativeBinomial as TorchNegativeBinomial

from uni2ts.distribution.negative_binomial import (
    NegativeBinomial,
    NegativeBinomialOutput,
)

NB_PARAMS = [
    (
        torch.as_tensor(0.1),
        torch.as_tensor(0.0),
    ),
    (
        torch.as_tensor(1.0),
        torch.as_tensor(0.0),
    ),
    (
        torch.as_tensor([1.0, 2.0]),
        torch.as_tensor([2.3, 4.5]),
    ),
    (
        torch.as_tensor(
            [
                [1.5, 2.5],
                [3.5, 4.5],
            ]
        ),
        torch.as_tensor(
            [
                [2.3, 4.5],
                [6.7, 8.9],
            ]
        ),
    ),
]


@pytest.mark.parametrize("total_count, logits", NB_PARAMS)
@pytest.mark.parametrize("sample_shape", [(100_000,), (50_000, 2)])
def test_neg_binom(
    total_count: torch.Tensor,
    logits: torch.Tensor,
    sample_shape: tuple[int, ...],
):
    torch_nb = TorchNegativeBinomial(total_count, logits=logits)
    nb = NegativeBinomial(total_count, logits=logits)

    assert torch_nb.batch_shape == nb.batch_shape
    assert torch_nb.event_shape == nb.event_shape
    assert torch.allclose(torch_nb.mean, nb.mean)
    assert torch.allclose(torch_nb.variance, nb.variance)

    torch.manual_seed(0)
    sample = nb.sample(torch.Size(sample_shape))
    torch.manual_seed(0)
    torch_sample = torch_nb.sample(torch.Size(sample_shape))

    assert sample.shape == torch_sample.shape
    dims = tuple(i for i in range(len(sample_shape)))
    assert torch.allclose(
        sample.mean(dim=dims),
        torch_sample.mean(dim=dims),
        atol=1e-2,
    )
    assert torch.allclose(
        sample.var(dim=dims),
        torch_sample.var(dim=dims),
        atol=2e-1,
    )


@pytest.mark.parametrize("total_count, logits", NB_PARAMS)
@pytest.mark.parametrize(
    "value",
    [
        torch.as_tensor(0.0),
        torch.as_tensor(1.0),
        torch.as_tensor([0.0, 1.0]),
        torch.as_tensor(
            [
                [0.0, 1.0],
                [2.0, 3.0],
            ]
        ),
    ],
)
def test_neg_binom_log_prob(
    total_count: torch.Tensor,
    logits: torch.Tensor,
    value: torch.Tensor,
):
    nb = NegativeBinomial(total_count, logits=logits)
    torch_nb = TorchNegativeBinomial(total_count, logits=logits)
    log_prob = nb.log_prob(value)
    torch_log_prob = torch_nb.log_prob(value)
    assert torch.allclose(log_prob, torch_log_prob)


@pytest.mark.parametrize("total_count, logits", NB_PARAMS)
@pytest.mark.parametrize(
    "value",
    [
        torch.as_tensor(0.1),
        torch.as_tensor(1.2),
        torch.as_tensor([0.3, 1.4]),
        torch.as_tensor(
            [
                [0.5, 1.2],
                [2.6, 3.3],
            ]
        ),
    ],
)
def test_continuous_neg_binom_log_prob(
    total_count: torch.Tensor,
    logits: torch.Tensor,
    value: torch.Tensor,
):
    nb = NegativeBinomial(total_count, logits=logits, validate_args=True)
    log_prob = nb.log_prob(value)
    assert torch.isfinite(log_prob).all()
