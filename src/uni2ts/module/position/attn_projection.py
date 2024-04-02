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

import abc
import math
from functools import cached_property
from typing import Any, Optional

import torch
from einops import einsum, rearrange, repeat
from jaxtyping import Float, Int
from torch import nn


class Projection(nn.Module, abc.ABC):
    def __init__(self, proj_width: int, num_heads: int, num_groups: int, **kwargs: Any):
        super().__init__()
        self.proj_width = proj_width
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.heads_per_group = num_heads // num_groups

    @abc.abstractmethod
    def forward(
        self,
        x: Float[torch.Tensor, "*batch group hpg seq dim"],
        seq_id: Optional[Int[torch.Tensor, "*batch #group #hpg seq"]],
    ) -> Float[torch.Tensor, "*batch group hpg seq dim"]: ...


class IdentityProjection(Projection):
    def __init__(self, *, proj_width: int, num_heads: int, num_groups: int, **kwargs):
        super().__init__(proj_width, num_heads, num_groups)

    def forward(
        self,
        x: Float[torch.Tensor, "*batch group hpg seq dim"],
        seq_id: Optional[Int[torch.Tensor, "*batch #group #hpg seq"]] = None,
    ) -> Float[torch.Tensor, "*batch group hpg seq dim"]:
        return x


class RotaryProjection(Projection):
    def __init__(
        self,
        *,
        proj_width: int,
        num_heads: int,
        num_groups: int,
        max_len: int = 512,
        base: int = 10000,
    ):
        super().__init__(proj_width, num_heads, num_groups)
        assert (
            self.proj_width % 2 == 0
        ), f"proj_width must be even, got {self.proj_width}"
        self.register_buffer(
            "theta",
            1.0
            / torch.pow(
                base,
                torch.arange(0, self.proj_width, 2, dtype=torch.float)
                / self.proj_width,
            ),
            persistent=False,
        )
        self.register_buffer("cos", None, persistent=False)
        self.register_buffer("sin", None, persistent=False)
        self._init_freq(max_len=max_len)

    def _init_freq(self, max_len: int):
        if self.cos is None or self.cos.size(-2) < max_len:
            position = torch.arange(
                max_len, device=self.theta.device, dtype=self.theta.dtype
            )
            m_theta = einsum(position, self.theta, "length, width -> length width")
            m_theta = repeat(m_theta, "length width -> length (width 2)")
            self.register_buffer("cos", torch.cos(m_theta), persistent=False)
            self.register_buffer("sin", torch.sin(m_theta), persistent=False)

    @staticmethod
    def _rotate(x: Float[torch.Tensor, "... dim"]) -> Float[torch.Tensor, "... dim"]:
        x1, x2 = rearrange(x, "... (dim r) -> r ... dim", r=2)
        return rearrange([-x2, x1], "r ... dim -> ... (dim r)", r=2)  # noqa

    def forward(
        self,
        x: Float[torch.Tensor, "*batch group hpg seq dim"],
        seq_id: Optional[Int[torch.Tensor, "*batch #group #hpg seq"]],
    ) -> Float[torch.Tensor, "*batch group hpg seq dim"]:
        self._init_freq(max_len=seq_id.max() + 1)
        rot_cos = self.cos[seq_id]
        rot_sin = self.sin[seq_id]
        return rot_cos * x + rot_sin * self._rotate(x)


class LearnedProjection(Projection):
    def __init__(
        self,
        *,
        proj_width: int,
        num_heads: int,
        num_groups: int,
        max_len: int = 512,
    ):
        super().__init__(proj_width, num_heads, num_groups)
        self.max_len = max_len
        self.weight = nn.Parameter(
            torch.empty((max_len, self.proj_width, self.proj_width))
        )
        self.reset_parameters()

    def reset_parameters(self):
        for idx in range(self.max_len):
            nn.init.kaiming_uniform_(self.weight[idx], a=math.sqrt(5))

    def forward(
        self,
        x: Float[torch.Tensor, "*batch group hpg seq dim"],
        seq_id: Optional[Int[torch.Tensor, "*batch #group #hpg seq"]],
    ) -> Float[torch.Tensor, "*batch group hpg seq dim"]:
        weight = self.weight[seq_id]
        return einsum(weight, x, "... out inp, ... inp -> ... out")


class QueryKeyProjection(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_groups: int,
        proj_layer: type[Projection],
        kwargs: Optional[dict[str, Any]] = None,
        key_proj_layer: Optional[type[Projection]] = None,
        key_kwargs: Optional[dict[str, Any]] = None,
        partial_factor: Optional[tuple[float, float]] = None,
    ):
        super().__init__()
        if partial_factor is not None:
            assert (
                0.0 <= partial_factor[0] < partial_factor[1] <= 1.0
            ), f"got {partial_factor[0]}, {partial_factor[1]}"
        assert num_heads > 0 and dim % num_heads == 0
        assert (num_heads % num_groups == 0) and (num_heads >= num_groups)

        self.head_dim = dim // num_heads
        self.partial_factor = partial_factor
        self.query_proj = proj_layer(
            proj_width=self.proj_width,
            num_heads=num_heads,
            num_groups=num_groups,
            **(kwargs or {}),
        )
        if key_proj_layer is None:  # Same as query
            self.key_proj = self.query_proj
        else:
            self.key_proj = key_proj_layer(
                proj_width=self.proj_width,
                num_heads=num_heads,
                num_groups=num_groups,
                **(key_kwargs or {}),
            )

    @cached_property
    def proj_width(self) -> int:
        if self.partial_factor is None:
            return self.head_dim
        return int(self.head_dim * (self.partial_factor[1] - self.partial_factor[0]))

    @cached_property
    def split_sizes(self) -> tuple[int, int, int]:
        if self.partial_factor is None:
            return 0, self.head_dim, 0
        return (
            int(self.partial_factor[0] * self.head_dim),
            self.proj_width,
            int((1.0 - self.partial_factor[1]) * self.head_dim),
        )

    def forward(
        self,
        query: Float[torch.Tensor, "*batch group hpg q_len dim"],
        key: Float[torch.Tensor, "*batch group hpg kv_len dim"],
        query_id: Optional[Int[torch.Tensor, "*batch #group #hpg q_len"]],
        kv_id: Optional[Int[torch.Tensor, "*batch #group #hpg kv_len"]],
    ) -> tuple[
        Float[torch.Tensor, "*batch group hpg seq dim"],
        Float[torch.Tensor, "*batch group hpg seq dim"],
    ]:
        # Split query/key into segments. Apply projections to the central segment (as determined by partial_factor)
        # Then concatenate the segments back together.
        if self.partial_factor is not None:
            queries = list(query.split(self.split_sizes, dim=-1))
            keys = list(key.split(self.split_sizes, dim=-1))
            queries[1] = self.query_proj(queries[1], seq_id=query_id)
            keys[1] = self.key_proj(keys[1], seq_id=kv_id)
            query = torch.cat(queries, dim=-1)
            key = torch.cat(keys, dim=-1)
        else:
            query = self.query_proj(query, seq_id=query_id)
            key = self.key_proj(key, seq_id=kv_id)
        return query, key
