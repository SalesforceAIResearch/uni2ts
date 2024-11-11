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
from typing import Optional

import torch
import torch.nn.functional as F
from jaxtyping import Bool, Float, Int
from torch import nn

from .attention import GroupedQueryAttention
from .ffn import FeedForward, GatedLinearUnitFeedForward, MoEFeedForward
from .position import AttentionBias, QueryKeyProjection


class TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        self_attn: GroupedQueryAttention,
        ffn: FeedForward,
        norm1: Optional[nn.Module],
        norm2: Optional[nn.Module],
        post_attn_dropout_p: float = 0.0,
        pre_norm: bool = True,
    ):
        super().__init__()
        self.pre_norm = pre_norm
        self.dropout_p = post_attn_dropout_p

        self.self_attn = self_attn
        self.ffn = ffn
        self.norm1 = norm1 or nn.Identity()
        self.norm2 = norm2 or nn.Identity()
        self.dropout = nn.Dropout(post_attn_dropout_p)

    def forward(
        self,
        x: Float[torch.Tensor, "*batch time_len dim"],
        attn_mask: Optional[Bool[torch.Tensor, "*batch time_len time_len"]] = None,
        var_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
        time_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
        centroid: Optional[Float[torch.Tensor, "expert dim"]] = None,
    ) -> Float[torch.Tensor, "*batch time_len dim"]:
        if self.pre_norm:
            x = x + self._sa_block(
                self.norm1(x), attn_mask, var_id=var_id, time_id=time_id
            )
            x = x + self.ffn(self.norm2(x), centroid=centroid)
        else:
            x = self.norm1(
                x + self._sa_block(x, attn_mask, var_id=var_id, time_id=time_id)
            )
            x = self.norm2(x + self.ffn(x, centroid=centroid))

        return x

    def _sa_block(
        self,
        x: Float[torch.Tensor, "*batch time_len dim"],
        attn_mask: Optional[Bool[torch.Tensor, "*batch time_len time_len"]],
        var_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
        time_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
    ) -> Float[torch.Tensor, "*batch time_len dim"]:
        x = self.self_attn(
            x,
            x,
            x,
            attn_mask=attn_mask,
            query_var_id=var_id,
            kv_var_id=var_id,
            query_time_id=time_id,
            kv_time_id=time_id,
        )
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_layers: int,
        num_heads: Optional[int] = None,
        num_groups: Optional[int] = None,
        pre_norm: bool = True,
        attn_dropout_p: float = 0.0,
        dropout_p: float = 0.0,
        norm_layer: Optional[Callable[[int], nn.Module]] = nn.LayerNorm,
        activation: Callable[[torch.Tensor], torch.Tensor] = F.silu,
        use_moe: bool = False,
        use_glu: bool = True,
        use_qk_norm: bool = True,
        var_attn_bias_layer: Optional[Callable[[int, int, int], AttentionBias]] = None,
        time_attn_bias_layer: Optional[Callable[[int, int, int], AttentionBias]] = None,
        var_qk_proj_layer: Optional[
            Callable[[int, int, int], QueryKeyProjection]
        ] = None,
        time_qk_proj_layer: Optional[
            Callable[[int, int, int], QueryKeyProjection]
        ] = None,
        shared_var_attn_bias: bool = False,
        shared_time_attn_bias: bool = False,
        shared_var_qk_proj: bool = False,
        shared_time_qk_proj: bool = False,
        d_ff: Optional[int] = None,
    ):
        super().__init__()
        self.use_moe = use_moe
        num_heads = num_heads or d_model // 64
        num_groups = num_groups or num_heads  # defaults to mha

        var_attn_bias = self.get_layer(
            d_model,
            num_heads,
            num_groups,
            var_attn_bias_layer,
            shared_var_attn_bias,
        )
        time_attn_bias = self.get_layer(
            d_model,
            num_heads,
            num_groups,
            time_attn_bias_layer,
            shared_time_attn_bias,
        )
        var_qk_proj = self.get_layer(
            d_model, num_heads, num_groups, var_qk_proj_layer, shared_var_qk_proj
        )
        time_qk_proj = self.get_layer(
            d_model, num_heads, num_groups, time_qk_proj_layer, shared_time_qk_proj
        )

        get_self_attn = partial(
            GroupedQueryAttention,
            dim=d_model,
            num_heads=num_heads,
            num_groups=num_groups,
            bias=False,
            norm_layer=norm_layer if use_qk_norm else None,
            softmax_scale=None,
            attn_dropout_p=attn_dropout_p,
            var_attn_bias=var_attn_bias,
            time_attn_bias=time_attn_bias,
            var_qk_proj=var_qk_proj,
            time_qk_proj=time_qk_proj,
        )
        if not use_moe:
            get_ffn = partial(
                GatedLinearUnitFeedForward if use_glu else FeedForward,
                in_dim=d_model,
                hidden_dim=d_ff,
                out_dim=None,
                activation=activation,
                bias=False,
                ffn_dropout_p=dropout_p,
            )
        else:
            get_ffn = partial(
                MoEFeedForward,
                num_experts=32,
                num_experts_per_token=2,
                in_dim=d_model,
                hidden_dim=d_ff,
                out_dim=None,
                activation=activation,
                bias=False,
                ffn_dropout_p=dropout_p,
            )
            self.register_buffer(
                "centroid", torch.empty(num_layers, 32, d_model, dtype=torch.float64)
            )
        get_encoder_layer_norm = partial(norm_layer, d_model)

        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    self_attn=get_self_attn(),
                    ffn=get_ffn(),
                    norm1=get_encoder_layer_norm(),
                    norm2=get_encoder_layer_norm(),
                    pre_norm=pre_norm,
                    post_attn_dropout_p=dropout_p,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm = norm_layer(d_model)

    @staticmethod
    def get_layer(
        dim: int,
        num_heads: int,
        num_groups: int,
        layer: Callable,
        shared_layer: bool,
    ) -> Optional[Callable[[], nn.Module]]:
        if layer is None:
            return None
        if shared_layer:
            module = layer(dim=dim, num_heads=num_heads, num_groups=num_groups)
            return lambda: module
        return partial(layer, dim=dim, num_heads=num_heads, num_groups=num_groups)

    def forward(
        self,
        x: Float[torch.Tensor, "*batch time_len dim"],
        attn_mask: Optional[Bool[torch.Tensor, "*batch time_len time_len"]] = None,
        var_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
        time_id: Optional[Int[torch.Tensor, "*batch time_len"]] = None,
    ) -> Float[torch.Tensor, "*batch time_len dim"]:
        if self.use_moe:
            for idx, layer in enumerate(self.layers):
                x = layer(
                    x,
                    attn_mask,
                    var_id=var_id,
                    time_id=time_id,
                    centroid=self.centroid[idx],
                )
        else:
            for layer in self.layers:
                x = layer(x, attn_mask, var_id=var_id, time_id=time_id)
        return self.norm(x)
