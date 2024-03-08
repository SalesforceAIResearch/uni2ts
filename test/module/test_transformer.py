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
from contextlib import nullcontext as does_not_raise
from functools import partial
from typing import ContextManager, Optional

import pytest
import torch
import torch.nn.functional as F
from einops import repeat
from torch import nn

from uni2ts.module.attention import GroupedQueryAttention
from uni2ts.module.ffn import FeedForward
from uni2ts.module.position import (
    AttentionBias,
    BinaryAttentionBias,
    IdentityProjection,
    LearnedEmbedding,
    LearnedProjection,
    LinearAttentionBias,
    Projection,
    QueryKeyProjection,
    RelativeAttentionBias,
    RotaryProjection,
    SinusoidalPositionEncoding,
)
from uni2ts.module.transformer import TransformerEncoder, TransformerEncoderLayer


@pytest.mark.parametrize("post_attn_dropout_p", [0.0, 0.1])
@pytest.mark.parametrize("pre_norm", [False, True])
def test_transformer_encoder_layer(
    post_attn_dropout_p: float,
    pre_norm: bool,
    dim: int = 64,
    norm_layer: type[nn.Module] = nn.LayerNorm,
):
    encoder_layer = TransformerEncoderLayer(
        GroupedQueryAttention(
            dim=dim,
            num_heads=1,
            num_groups=1,
            bias=True,
            norm_layer=norm_layer,
            softmax_scale=None,
            attn_dropout_p=0.0,
        ),
        FeedForward(
            in_dim=dim,
            activation=F.gelu,
            bias=False,
            ffn_dropout_p=0.0,
        ),
        norm1=norm_layer(dim),
        norm2=norm_layer(dim),
        post_attn_dropout_p=post_attn_dropout_p,
        pre_norm=pre_norm,
    )
    x = torch.randn(2, 3, dim)
    y = encoder_layer(x)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()


@pytest.mark.parametrize("batch_shape", [(1,), (2, 3)])
@pytest.mark.parametrize(
    "d_model, num_layers, num_heads, expectation",
    [
        (384, 1, 6, does_not_raise()),
        (512, 3, 8, does_not_raise()),
        (384, 1, None, does_not_raise()),
        (63, 1, None, pytest.raises(AssertionError)),
    ],
)
@pytest.mark.parametrize("pre_norm", [False, True])
@pytest.mark.parametrize("use_glu", [False, True])
@pytest.mark.parametrize("num_groups", [None, 2])
@pytest.mark.parametrize("d_ff", [None, 128])
def test_transformer_encoder(
    batch_shape: tuple[int, ...],
    d_model: int,
    num_layers: int,
    num_heads: int,
    pre_norm: bool,
    use_glu: bool,
    num_groups: Optional[int],
    d_ff: Optional[int],
    expectation: ContextManager,
    dropout_p: float = 0.1,
):
    with expectation:
        encoder = TransformerEncoder(
            d_model,
            num_layers,
            num_heads=num_heads,
            pre_norm=pre_norm,
            dropout_p=dropout_p,
            use_glu=use_glu,
            num_groups=num_groups,
            d_ff=d_ff,
        )
        x = torch.randn(batch_shape + (d_model,))
        y = encoder(x)
        assert y.shape == x.shape
        assert not torch.isnan(y).any()


@pytest.mark.parametrize(
    "var_attn_bias_layer",
    [
        None,
        BinaryAttentionBias,
    ],
)
@pytest.mark.parametrize(
    "time_attn_bias_layer",
    [
        None,
        BinaryAttentionBias,
    ],
)
@pytest.mark.parametrize(
    "var_qk_proj_layer",
    [
        None,
        partial(
            QueryKeyProjection,
            proj_layer=IdentityProjection,
        ),
        partial(
            QueryKeyProjection,
            proj_layer=RotaryProjection,
            kwargs=dict(max_len=512),
            partial_factor=(0.0, 0.5),
        ),
        partial(
            QueryKeyProjection,
            proj_layer=RotaryProjection,
            key_proj_layer=LearnedProjection,
        ),
    ],
)
@pytest.mark.parametrize(
    "time_qk_proj_layer",
    [
        None,
        partial(
            QueryKeyProjection,
            proj_layer=IdentityProjection,
        ),
        partial(
            QueryKeyProjection,
            proj_layer=RotaryProjection,
            kwargs=dict(max_len=512),
            partial_factor=(0.0, 0.5),
        ),
        partial(
            QueryKeyProjection,
            proj_layer=RotaryProjection,
            key_proj_layer=LearnedProjection,
        ),
    ],
)
@pytest.mark.parametrize("shared_layers", [True, False])
def test_transformer_encoder_position_init(
    var_attn_bias_layer: Optional[Callable[[int, int, int], AttentionBias]],
    time_attn_bias_layer: Optional[Callable[[int, int, int], AttentionBias]],
    var_qk_proj_layer: Optional[Callable[[int, int, int], QueryKeyProjection]],
    time_qk_proj_layer: Optional[Callable[[int, int, int], QueryKeyProjection]],
    shared_layers: bool,
    d_model: int = 384,
    num_layers: int = 2,
):
    encoder = TransformerEncoder(
        d_model,
        num_layers,
        num_heads=None,
        pre_norm=True,
        use_glu=False,
        num_groups=None,
        var_attn_bias_layer=var_attn_bias_layer,
        time_attn_bias_layer=time_attn_bias_layer,
        var_qk_proj_layer=var_qk_proj_layer,
        time_qk_proj_layer=time_qk_proj_layer,
        shared_var_attn_bias=shared_layers,
        shared_time_attn_bias=shared_layers,
        shared_time_qk_proj=shared_layers,
        shared_var_qk_proj=shared_layers,
    )

    if shared_layers:
        if var_attn_bias_layer is not None:
            assert (
                encoder.layers[0].self_attn.var_attn_bias
                == encoder.layers[1].self_attn.var_attn_bias
            )
        if time_attn_bias_layer is not None:
            assert (
                encoder.layers[0].self_attn.time_attn_bias
                == encoder.layers[1].self_attn.time_attn_bias
            )
        if var_qk_proj_layer is not None:
            assert (
                encoder.layers[0].self_attn.var_qk_proj
                == encoder.layers[1].self_attn.var_qk_proj
            )
        if time_qk_proj_layer is not None:
            assert (
                encoder.layers[0].self_attn.time_qk_proj
                == encoder.layers[1].self_attn.time_qk_proj
            )
    else:
        if var_attn_bias_layer is not None:
            assert (
                encoder.layers[0].self_attn.var_attn_bias
                != encoder.layers[1].self_attn.var_attn_bias
            )
        if time_attn_bias_layer is not None:
            assert (
                encoder.layers[0].self_attn.time_attn_bias
                != encoder.layers[1].self_attn.time_attn_bias
            )
        if var_qk_proj_layer is not None:
            assert (
                encoder.layers[0].self_attn.var_qk_proj
                != encoder.layers[1].self_attn.var_qk_proj
            )
        if time_qk_proj_layer is not None:
            assert (
                encoder.layers[0].self_attn.time_qk_proj
                != encoder.layers[1].self_attn.time_qk_proj
            )


@pytest.mark.parametrize(
    "var_attn_bias_layer",
    [
        None,
        BinaryAttentionBias,
    ],
)
@pytest.mark.parametrize(
    "time_attn_bias_layer",
    [
        None,
        BinaryAttentionBias,
    ],
)
@pytest.mark.parametrize(
    "var_qk_proj_layer",
    [
        None,
        partial(
            QueryKeyProjection,
            proj_layer=RotaryProjection,
            kwargs=dict(max_len=512),
            partial_factor=(0.0, 0.5),
        ),
    ],
)
@pytest.mark.parametrize(
    "time_qk_proj_layer",
    [
        None,
        partial(
            QueryKeyProjection,
            proj_layer=RotaryProjection,
            kwargs=dict(max_len=512),
            partial_factor=(0.0, 0.5),
        ),
    ],
)
@pytest.mark.parametrize("batch_shape", [tuple(), (1,), (2, 3)])
@pytest.mark.parametrize("use_attn_mask", [True, False])
@pytest.mark.parametrize("use_var_id", [True, False])
@pytest.mark.parametrize("use_time_id", [True, False])
def test_transformer_encoder_position_forward(
    var_attn_bias_layer: Optional[Callable[[int, int, int], AttentionBias]],
    time_attn_bias_layer: Optional[Callable[[int, int, int], AttentionBias]],
    var_qk_proj_layer: Optional[Callable[[int, int, int], QueryKeyProjection]],
    time_qk_proj_layer: Optional[Callable[[int, int, int], QueryKeyProjection]],
    batch_shape: tuple[int, ...],
    use_attn_mask: bool,
    use_var_id: bool,
    use_time_id: bool,
    time_len: int = 5,
    d_model: int = 384,
    num_layers: int = 2,
    shared_layers: bool = True,
):
    encoder = TransformerEncoder(
        d_model,
        num_layers,
        num_heads=None,
        pre_norm=True,
        use_glu=False,
        num_groups=None,
        var_attn_bias_layer=var_attn_bias_layer,
        time_attn_bias_layer=time_attn_bias_layer,
        var_qk_proj_layer=var_qk_proj_layer,
        time_qk_proj_layer=time_qk_proj_layer,
        shared_var_attn_bias=shared_layers,
        shared_time_attn_bias=shared_layers,
        shared_time_qk_proj=shared_layers,
        shared_var_qk_proj=shared_layers,
    )

    x = torch.randn(
        batch_shape
        + (
            time_len,
            d_model,
        )
    )
    attn_mask = (
        torch.ones(batch_shape + (time_len, time_len), dtype=torch.bool)
        if use_attn_mask
        else None
    )
    var_id = (
        torch.zeros(batch_shape + (time_len,), dtype=torch.long) if use_var_id else None
    )
    time_id = (
        repeat(
            torch.arange(time_len, dtype=torch.long),
            f"seq -> {' '.join(map(str, batch_shape))} seq",
        )
        if use_time_id
        else None
    )
    y = encoder(x, attn_mask=attn_mask, var_id=var_id, time_id=time_id)
    assert y.shape == x.shape
    assert not torch.isnan(y).any()
