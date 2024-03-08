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

from itertools import product

import pytest
import torch
from einops import repeat

from uni2ts.module.position.attn_projection import (
    IdentityProjection,
    LearnedProjection,
    QueryKeyProjection,
    RotaryProjection,
)


@pytest.mark.parametrize(
    "batch_shape, dim",
    [
        (tuple(), 4 * 8),
        (tuple(), 8 * 8),
        ((2,), 16 * 8),
        ((2, 3), 32 * 8),
    ],
)
@pytest.mark.parametrize(
    "seq_len",
    [5, 10, 11],
)
def test_rotary_projection(
    batch_shape: tuple[int, ...],
    seq_len: int,
    dim: int,
    num_heads: int = 8,
    num_groups: int = 8,
    max_len: int = 10,
    seed: int = 0,
):
    torch.manual_seed(seed)
    rotary = RotaryProjection(
        proj_width=dim, num_heads=num_heads, num_groups=num_groups, max_len=max_len
    )

    batch_shape_str = " ".join(str(bs) for bs in batch_shape)
    x = torch.randn(batch_shape + (seq_len, rotary.proj_width))
    pos_id = repeat(
        torch.arange(seq_len),
        f"seq_len -> {batch_shape_str} seq_len",
    )

    def get_square(theta_i, m):
        return torch.as_tensor(
            [
                [torch.cos(m * theta_i), -torch.sin(m * theta_i)],
                [torch.sin(m * theta_i), torch.cos(m * theta_i)],
            ]
        )

    R = torch.zeros(batch_shape + (seq_len, rotary.proj_width, rotary.proj_width))
    for idx in product(*[range(x) for x in batch_shape + (seq_len,)]):
        m = pos_id[idx].float()
        for w_id in range(1, rotary.proj_width // 2 + 1):
            theta_i = 10000 ** (-2 * (w_id - 1) / rotary.proj_width)
            w_id -= 1
            R[idx][2 * w_id : 2 * w_id + 2, 2 * w_id : 2 * w_id + 2] = get_square(
                theta_i, m
            )

    Rx = (R @ x.unsqueeze(-1)).squeeze(-1)

    rot_x = rotary(x, seq_id=pos_id)
    assert rot_x.shape == batch_shape + (seq_len, rotary.proj_width)
    assert torch.allclose(rot_x, Rx, atol=1e-6)
