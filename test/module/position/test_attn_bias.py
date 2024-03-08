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

from uni2ts.module.position.attn_bias import (
    AttentionBias,
    BinaryAttentionBias,
    LinearAttentionBias,
    RelativeAttentionBias,
)


@pytest.mark.parametrize("batch_shape", [tuple(), (1,), (2, 3)])
@pytest.mark.parametrize("num_heads, num_groups", [(8, 1), (8, 2), (8, 8)])
def test_binary_attention_bias(
    batch_shape: tuple[int, ...],
    num_heads: int,
    num_groups: int,
    q_len: int = 5,
    kv_len: int = 3,
    dim: int = 64,
):
    attn_bias = BinaryAttentionBias(dim=dim, num_heads=num_heads, num_groups=num_groups)
    bias = attn_bias(
        None,
        None,
        query_id=torch.randint(2, size=(*batch_shape, 1, 1, q_len)),
        kv_id=torch.randint(2, size=(*batch_shape, 1, 1, kv_len)),
    )
    assert bias.shape == (
        *batch_shape,
        num_groups,
        num_heads // num_groups,
        q_len,
        kv_len,
    )


@pytest.mark.parametrize("batch_shape", [tuple(), (1,), (2, 3)])
@pytest.mark.parametrize("num_heads, num_groups", [(8, 1), (8, 2), (8, 8)])
def test_linear_attention_bias(
    batch_shape: tuple[int, ...],
    num_heads: int,
    num_groups: int,
    q_len: int = 5,
    kv_len: int = 3,
    dim: int = 64,
):
    attn_bias = LinearAttentionBias(dim=dim, num_heads=num_heads, num_groups=num_groups)
    bias = attn_bias(
        None,
        None,
        query_id=torch.randint(2, size=(*batch_shape, 1, 1, q_len)),
        kv_id=torch.randint(2, size=(*batch_shape, 1, 1, kv_len)),
    )
    assert bias.shape == (
        *batch_shape,
        num_groups,
        num_heads // num_groups,
        q_len,
        kv_len,
    )
