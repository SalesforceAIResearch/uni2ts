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
from typing import Optional

import pytest
import torch
from torch import nn

from uni2ts.module.attention import (
    GroupedQueryAttention,
    MultiHeadAttention,
    MultiQueryAttention,
)


@pytest.mark.parametrize("batch_shape", [tuple(), (2,)])
@pytest.mark.parametrize("dim", [64, 384])
@pytest.mark.parametrize("kv_len", [1, 10])
@pytest.mark.parametrize("q_len", [10])
@pytest.mark.parametrize("use_attn_mask", [False, True])
@pytest.mark.parametrize(
    "num_heads, num_groups",
    [
        (8, 8),
        (8, 1),
        (8, 2),
    ],
)
@pytest.mark.parametrize(
    "norm_layer", [None, nn.LayerNorm, partial(nn.LayerNorm, bias=False)]
)
@pytest.mark.parametrize("softmax_scale", [None, 0.1])
@pytest.mark.parametrize("attn_dropout_p", [0.15])
def test_gqa(
    batch_shape: tuple[int, ...],
    dim: int,
    kv_len: int,
    q_len: int,
    use_attn_mask: bool,
    num_heads: int,
    num_groups: int,
    norm_layer: Optional[nn.Module],
    softmax_scale: Optional[float],
    attn_dropout_p: float,
):
    attn = GroupedQueryAttention(
        dim=dim,
        num_heads=num_heads,
        num_groups=num_groups,
        bias=True,
        norm_layer=norm_layer,
        softmax_scale=softmax_scale,
        attn_dropout_p=0.0,
    )

    query = torch.randn(*(batch_shape + (q_len, dim)))
    key = torch.randn(*(batch_shape + (kv_len, dim)))
    value = torch.randn(*(batch_shape + (kv_len, dim)))

    attn_mask = (
        torch.ones(*(batch_shape + (q_len, kv_len)), dtype=torch.bool)
        if use_attn_mask
        else None
    )

    out = attn(query, key, value, attn_mask=attn_mask)
    assert out.shape == batch_shape + (q_len, dim)
    assert not out.isnan().any()


def test_all_masked(
    batch_shape: tuple[int, ...] = tuple(),
    dim: int = 384,
    q_len: int = 10,
    kv_len: int = 10,
):
    attn = GroupedQueryAttention(
        dim=dim,
        num_heads=8,
        num_groups=2,
        bias=True,
    )

    query = torch.randn(*(batch_shape + (q_len, dim)))
    key = torch.randn(*(batch_shape + (kv_len, dim)))
    value = torch.randn(*(batch_shape + (kv_len, dim)))

    attn_mask = torch.zeros(*(batch_shape + (q_len, kv_len)), dtype=torch.bool)
    out = attn(query, key, value, attn_mask=attn_mask)
    assert torch.isnan(out).all()


def test_mqa(
    batch_shape: tuple[int, ...] = (2,),
    dim: int = 384,
    q_len: int = 10,
    kv_len: int = 10,
):
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(
        dim=dim,
        num_heads=6,
        num_groups=1,
        bias=True,
        norm_layer=None,
        softmax_scale=None,
        attn_dropout_p=0.0,
    )
    torch.manual_seed(0)
    mqa = MultiQueryAttention(
        dim=dim,
        num_heads=6,
        bias=True,
        norm_layer=None,
        softmax_scale=None,
        attn_dropout_p=0.0,
    )
    query = torch.randn(*(batch_shape + (q_len, dim)))
    key = torch.randn(*(batch_shape + (kv_len, dim)))
    value = torch.randn(*(batch_shape + (kv_len, dim)))

    assert torch.eq(gqa(query, key, value), mqa(query, key, value)).all()


def test_mha(
    batch_shape: tuple[int, ...] = (2,),
    dim: int = 384,
    q_len: int = 10,
    kv_len: int = 10,
):
    torch.manual_seed(0)
    gqa = GroupedQueryAttention(
        dim=dim,
        num_heads=6,
        num_groups=6,
        bias=True,
        norm_layer=None,
        softmax_scale=None,
        attn_dropout_p=0.0,
    )
    torch.manual_seed(0)
    mha = MultiHeadAttention(
        dim=dim,
        num_heads=6,
        bias=True,
        norm_layer=None,
        softmax_scale=None,
        attn_dropout_p=0.0,
    )
    query = torch.randn(*(batch_shape + (q_len, dim)))
    key = torch.randn(*(batch_shape + (kv_len, dim)))
    value = torch.randn(*(batch_shape + (kv_len, dim)))

    assert torch.eq(gqa(query, key, value), mha(query, key, value)).all()
