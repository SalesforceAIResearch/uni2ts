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

import math

import pytest
import torch

from uni2ts.module.norm import RMSNorm


@pytest.mark.parametrize("batch_shape", [tuple(), (1,), (3, 2)])
@pytest.mark.parametrize(
    "normalized_shape", [32, (32,), (32, 64), torch.Size([32]), torch.Size([32, 64])]
)
@pytest.mark.parametrize("weight", [True, False])
def test_rms_norm(
    batch_shape: tuple[int, ...],
    normalized_shape: int | list[int, ...] | torch.Size,
    weight: bool,
):
    tupled_normalized_shape = (
        (normalized_shape,) if isinstance(normalized_shape, int) else normalized_shape
    )
    norm = RMSNorm(normalized_shape, weight=weight)
    x = torch.randn(batch_shape + tupled_normalized_shape)
    x_normed = norm(x)

    def rms_norm(inp: torch.Tensor) -> torch.Tensor:
        norm_inp = torch.linalg.vector_norm(
            inp, ord=2, dim=tuple(range(-len(tupled_normalized_shape), 0)), keepdim=True
        )
        rms_inp = norm_inp * math.prod(tupled_normalized_shape) ** (-1.0 / 2)
        inp_normed = inp / (rms_inp + 1e-5)
        return inp_normed

    assert x_normed.shape == x.shape
    assert torch.allclose(x_normed, rms_norm(x))
    if weight:
        assert norm.weight.shape == tupled_normalized_shape
        assert torch.allclose(norm.weight, torch.ones(tupled_normalized_shape))
    else:
        assert norm.weight is None
