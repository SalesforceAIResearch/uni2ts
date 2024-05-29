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

import numpy as np
import pytest
import torch
from einops import einsum, rearrange
from jaxtyping import PyTree
from torch.utils._pytree import tree_flatten, tree_map

from uni2ts.distribution import DistrParamProj
from uni2ts.distribution._base import convert_to_container, tree_map_multi
from uni2ts.module.ts_embed import MultiOutSizeLinear


@pytest.mark.parametrize(
    "out_features",
    [
        (1,),
        [1],
        (8,),
        [8],
        (8, 16, 32),
        [8, 16, 32],
    ],
)
@pytest.mark.parametrize(
    "args_dim",
    [
        {"loc": 1, "scale": 1, "df": 1},
        {"loc": 2, "scale": 1, "df": 1},
        {"weights_logit": 2, "components": [{"loc1": 1}, {"loc2": 1, "scale2": 1}]},
        {
            "weights_logit": 4,
            "components": [
                {"loc1": 1},
                {"loc2": 1},
                {"loc3": 1},
                {"loc4": 1, "scale4": 1},
            ],
        },
    ],
)
@pytest.mark.parametrize("batch_shape", [tuple(), (100,), (100, 10)])
def test_multi_out_size_linear_proj(
    out_features: tuple[int, ...] | list[int],
    args_dim: PyTree[int],
    batch_shape: tuple[int, ...],
    in_features: int = 32,
):
    out_proj = DistrParamProj(
        in_features=in_features,
        out_features=out_features,
        args_dim=args_dim,
        domain_map=tree_map(lambda x: lambda y: y, args_dim),
        proj_layer=MultiOutSizeLinear,
    )

    x = torch.randn(*batch_shape, in_features)
    out_feat_size = torch.as_tensor(np.random.choice(out_features, batch_shape))
    out = out_proj(x, out_feat_size)

    for feat_size in out_features:
        feat_size_x = x[out_feat_size == feat_size]

        def check_shape(out_leaf: torch.Tensor, proj: MultiOutSizeLinear) -> bool:
            feat_size_out = out_leaf[out_feat_size == feat_size]
            try:
                feat_size_weight = proj.weight[
                    proj.out_features_ls.index(feat_size * proj.dim)
                ]
            except ValueError as e:
                print(proj.out_features_ls, feat_size, proj.dim, feat_size * proj.dim)
                raise e
            feat_size_bias = (
                proj.bias[proj.out_features_ls.index(feat_size * proj.dim)]
                if proj.bias is not None
                else 0
            )
            feat_size_gt = rearrange(
                einsum(feat_size_weight, feat_size_x, "out inp, ... inp -> ... out")
                + feat_size_bias,
                "... (dim out_size) -> ... out_size dim",
                out_size=max(out_features),
            )
            return feat_size_gt.shape == feat_size_out.shape

        def check_all_close(out_leaf: torch.Tensor, proj: MultiOutSizeLinear) -> bool:
            feat_size_out = out_leaf[out_feat_size == feat_size]
            try:
                feat_size_weight = proj.weight[
                    proj.out_features_ls.index(feat_size * proj.dim)
                ]
            except ValueError as e:
                print(proj.out_features_ls, feat_size, proj.dim, feat_size * proj.dim)
                raise e
            feat_size_bias = (
                proj.bias[proj.out_features_ls.index(feat_size * proj.dim)]
                if proj.bias is not None
                else 0
            )
            feat_size_gt = rearrange(
                einsum(feat_size_weight, feat_size_x, "out inp, ... inp -> ... out")
                + feat_size_bias,
                "... (dim out_size) -> ... out_size dim",
                out_size=max(out_features),
            )
            return torch.allclose(feat_size_gt, feat_size_out, atol=1e-6)

        assert all(
            tree_flatten(
                tree_map_multi(check_shape, out, convert_to_container(out_proj.proj))
            )[0]
        )

        assert all(
            tree_flatten(
                tree_map_multi(
                    check_all_close, out, convert_to_container(out_proj.proj)
                )
            )[0]
        )
