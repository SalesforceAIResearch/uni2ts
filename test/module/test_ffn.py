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

from typing import Callable, Optional

import pytest
import torch
import torch.nn.functional as F

from uni2ts.module.ffn import FeedForward, GatedLinearUnitFeedForward


@pytest.mark.parametrize("use_glu", [False, True])
@pytest.mark.parametrize("batch_shape", [(), (2,), (2, 3)])
@pytest.mark.parametrize(
    "in_dim, hidden_dim, out_dim",
    [
        (64, None, None),
        (64, 128, None),
        (64, None, 128),
        (64, 128, 128),
    ],
)
@pytest.mark.parametrize("activation", [F.relu, F.silu, F.gelu])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("ffn_dropout_p", [0.0, 0.1])
def test_feedforward(
    use_glu: bool,
    batch_shape: tuple[int, ...],
    in_dim: int,
    hidden_dim: Optional[int],
    out_dim: Optional[int],
    activation: Callable[[torch.Tensor], torch.Tensor],
    bias: bool,
    ffn_dropout_p: float,
):
    ffn_cls = GatedLinearUnitFeedForward if use_glu else FeedForward
    ffn = ffn_cls(
        in_dim, hidden_dim, out_dim, activation, bias=bias, ffn_dropout_p=ffn_dropout_p
    )
    x = torch.randn(batch_shape + (in_dim,))
    y = ffn(x)
    assert y.shape == x.shape[:-1] + (ffn.out_dim,)
    assert not torch.isnan(y).any()
