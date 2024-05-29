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
from einops import pack, unpack

from uni2ts.module.ts_embed import MultiInSizeLinear, MultiOutSizeLinear


@pytest.mark.parametrize("batch_shape", [tuple(), (1,), (10, 3)])
@pytest.mark.parametrize("in_features_ls", [(10,), (10, 20, 30)])
@pytest.mark.parametrize("out_features", [64, 128, 512])
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_multi_in_size_linear(
    batch_shape: tuple[int, ...],
    in_features_ls: tuple[int, ...],
    out_features: int,
    bias: bool,
    seed: int,
):
    # init data
    torch.manual_seed(seed)
    x = torch.randn(batch_shape + (max(in_features_ls),))
    inp_idx = torch.randint(0, len(in_features_ls), batch_shape)
    inp_ifs = torch.as_tensor(in_features_ls)[inp_idx]

    # init model
    torch.manual_seed(seed)
    embed = MultiInSizeLinear(
        in_features_ls,
        out_features,
        bias=bias,
    )
    embed_out = embed(x, inp_ifs)

    # init ground truth model
    torch.manual_seed(seed)
    linears = [
        torch.nn.Linear(in_features, out_features, bias=bias)
        for in_features in in_features_ls
    ]
    packed_x, ps = pack([x], "* max_feat")
    packed_inp_idx, _ = pack([inp_idx], "*")
    linears_out = torch.stack(
        [
            linears[packed_inp_idx[i]](packed_x[i, : in_features_ls[packed_inp_idx[i]]])
            for i in range(len(packed_inp_idx))
        ]
    )
    linears_out = unpack(linears_out, ps, "* out_feat")[0]

    # compute loss and backprop
    torch.mean((embed_out - linears_out.detach()) ** 2).backward()
    torch.mean((linears_out - embed_out.detach()) ** 2).backward()

    # check
    for idx in range(len(in_features_ls)):
        # check weights
        assert torch.allclose(
            embed.weight[idx, :, : in_features_ls[idx]], linears[idx].weight
        )
        assert embed.weight[idx, :, in_features_ls[idx] :].eq(0).all()
        if bias:
            assert torch.allclose(embed.bias[idx], linears[idx].bias)
        # check grads
        if idx in inp_idx:
            assert torch.allclose(
                embed.weight.grad[idx, :, : in_features_ls[idx]],
                linears[idx].weight.grad,
                atol=1e-6,
            )
            assert embed.weight.grad[idx, :, in_features_ls[idx] :].eq(0).all()
            if bias:
                assert torch.allclose(
                    embed.bias.grad[idx], linears[idx].bias.grad, atol=1e-6
                )
    # check output
    assert embed_out.shape == linears_out.shape == batch_shape + (out_features,)
    assert not embed_out.isnan().any().item()
    assert torch.allclose(embed_out, linears_out, atol=1e-6)


@pytest.mark.parametrize("batch_shape", [tuple(), (1,), (10, 3)])
@pytest.mark.parametrize("in_features", [64, 128, 512])
@pytest.mark.parametrize(
    "out_features_ls, dim",
    [
        ((10,), 1),
        ((20,), 2),
        ((10, 20, 30), 1),
        ((20, 40, 60), 2),
    ],
)
@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_multi_out_size_linear(
    batch_shape: tuple[int, ...],
    in_features: int,
    out_features_ls: tuple[int, ...],
    dim: int,
    bias: bool,
    seed: int,
):
    # init data
    torch.manual_seed(seed)
    x = torch.randn(batch_shape + (in_features,))
    inp_idx = torch.randint(0, len(out_features_ls), batch_shape)
    inp_ofs = torch.as_tensor(out_features_ls)[inp_idx]

    # init model
    torch.manual_seed(seed)
    embed = MultiOutSizeLinear(
        in_features,
        out_features_ls,
        dim=dim,
        bias=bias,
    )
    embed_out = embed(x, inp_ofs // dim)

    # init ground truth model
    torch.manual_seed(seed)
    linears = [
        torch.nn.Linear(in_features, out_features, bias=bias)
        for out_features in out_features_ls
    ]
    packed_x, ps = pack([x], "* max_feat")
    packed_inp_idx, _ = pack([inp_idx], "*")
    packed_inp_ofs, _ = pack([inp_ofs], "*")
    linears_out = torch.stack(
        [
            torch.cat(
                [
                    linears[packed_inp_idx[i]](packed_x[i]),
                    torch.zeros(max(out_features_ls) - packed_inp_ofs[i].item()),
                ]
            )
            for i in range(len(packed_inp_idx))
        ]
    )
    linears_out = unpack(linears_out, ps, "* out_feat")[0]

    # compute loss and backprop
    torch.mean((embed_out - linears_out.detach()) ** 2).backward()
    torch.mean((linears_out - embed_out.detach()) ** 2).backward()

    # check
    for idx in range(len(out_features_ls)):
        # check weights
        assert torch.allclose(
            embed.weight[idx, : out_features_ls[idx]], linears[idx].weight
        )
        assert embed.weight[idx, out_features_ls[idx] :].eq(0).all()
        if bias:
            assert torch.allclose(
                embed.bias[idx, : out_features_ls[idx]], linears[idx].bias
            )
            assert embed.bias[idx, out_features_ls[idx] :].eq(0).all()
        # check grads
        if idx in inp_idx:
            assert torch.allclose(
                embed.weight.grad[idx, : out_features_ls[idx]],
                linears[idx].weight.grad,
                atol=1e-6,
            )
            assert embed.weight.grad[idx, out_features_ls[idx] :].eq(0).all()
            if bias:
                assert torch.allclose(
                    embed.bias.grad[idx, : out_features_ls[idx]],
                    linears[idx].bias.grad,
                    atol=1e-6,
                )
                assert embed.bias.grad[idx, out_features_ls[idx] :].eq(0).all()
    # check output
    assert embed_out.shape == linears_out.shape == batch_shape + (max(out_features_ls),)
    assert not embed_out.isnan().any().item()
    assert torch.allclose(embed_out, linears_out, atol=1e-6)
