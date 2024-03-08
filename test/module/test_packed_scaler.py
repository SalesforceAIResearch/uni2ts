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

from typing import Callable

import pytest
import torch
from einops import rearrange, repeat

from uni2ts.common.torch_util import safe_div
from uni2ts.module.packed_scaler import (
    PackedAbsMeanScaler,
    PackedNOPScaler,
    PackedScaler,
    PackedStdScaler,
)


def pack_seq(
    xs: list[torch.Tensor],
    max_seq_len: int,
    pad_fn: Callable = torch.zeros,
):
    batch = [[]]
    shape = xs[0].shape[1:]
    for x in xs:
        if sum([b.shape[0] for b in batch[-1]]) + x.shape[0] > max_seq_len:
            batch.append([])
        batch[-1].append(x)

    for i in range(len(batch)):
        curr_len = sum([b.shape[0] for b in batch[i]])
        if curr_len < max_seq_len:
            batch[i].append(
                pad_fn(max_seq_len - curr_len, *shape, dtype=batch[i][0].dtype)
            )

    return torch.stack([torch.cat(x, dim=0) for x in batch], dim=0)


def _test_packed_scaler(
    get_loc_scale_func: Callable[
        [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ],
    packed_scaler: PackedScaler,
    samples: list[tuple[int, int, int, int, int]],
    max_seq_len: int,
    max_patch_size: int,
    observed_pct: float,
):
    data = {
        "target": [],
        "observed_mask": [],
        "sample_id": [],
        "dimension_id": [],
        "loc": [],
        "scale": [],
    }
    sample_id_counter = 1
    packing_counter = 0
    for time, n_target_time, target_dim, cov_dim, patch_size in samples:
        dim = target_dim + cov_dim
        assert time * dim <= max_seq_len

        _target = torch.randn(n_target_time, target_dim, patch_size)
        _cov = torch.randn(time, cov_dim, patch_size)
        _observed = torch.rand(time, dim, patch_size) < observed_pct

        target = torch.randn(time, dim, max_patch_size)
        target[-n_target_time:, :target_dim, :patch_size] = _target
        target[:, target_dim:, :patch_size] = _cov
        target = rearrange(target, "t d p -> (d t) p")

        observed_mask = torch.zeros(time, dim, max_patch_size, dtype=torch.bool)
        observed_mask[-n_target_time:, :target_dim, :patch_size] = True
        observed_mask[:, target_dim:, :patch_size] = True
        observed_mask[:, :, :patch_size] *= _observed
        observed_mask = rearrange(observed_mask, "t d p -> (d t) p")

        if packing_counter + time * dim > max_seq_len:
            sample_id_counter = 1
            packing_counter = 0
        else:
            sample_id_counter += 1
            packing_counter += time * dim
        sample_id = torch.ones(time * dim, dtype=torch.long) * sample_id_counter

        dimension_id = repeat(torch.arange(dim), "d -> (d t)", t=time)

        _target_loc, _target_scale = get_loc_scale_func(
            _target, _observed[-n_target_time:, :target_dim]
        )
        _cov_loc, _cov_scale = get_loc_scale_func(_cov, _observed[:, target_dim:])
        loc = repeat(
            torch.cat([_target_loc, _cov_loc], dim=1), "1 d -> (d t) 1", t=time
        )
        scale = repeat(
            torch.cat([_target_scale, _cov_scale], dim=1), "1 d -> (d t) 1", t=time
        )

        data["target"].append(target)
        data["observed_mask"].append(observed_mask)
        data["sample_id"].append(sample_id)
        data["dimension_id"].append(dimension_id)
        data["loc"].append(loc)
        data["scale"].append(scale)

    target = pack_seq(data["target"], max_seq_len)
    observed_mask = pack_seq(data["observed_mask"], max_seq_len)
    sample_id = pack_seq(data["sample_id"], max_seq_len)
    dimension_id = pack_seq(data["dimension_id"], max_seq_len)
    loc = pack_seq(data["loc"], max_seq_len)
    scale = pack_seq(data["scale"], max_seq_len, pad_fn=torch.ones)

    packed_loc, packed_scale = packed_scaler(
        target,
        observed_mask,
        sample_id,
        dimension_id,
    )

    assert loc.shape[0] == packed_loc.shape[0]
    assert loc.shape[1] == packed_loc.shape[1]
    assert torch.allclose(loc, packed_loc)
    assert scale.shape[0] == packed_scale.shape[0]
    assert scale.shape[1] == packed_scale.shape[1]
    assert torch.allclose(scale, packed_scale, atol=1e-4)


testdata = [
    (
        5,
        2,
        # time, n_target_time, target_dim, cov_dim, patch_size
        [(3, 1, 1, 0, 2), (2, 1, 1, 0, 1)],
    ),
    (
        20,
        3,
        [
            (3, 2, 1, 1, 1),
            (3, 1, 1, 1, 3),
            (3, 1, 1, 1, 3),
        ],
    ),
    (
        20,
        5,
        [
            (10, 5, 1, 0, 3),
            (10, 5, 1, 0, 3),
            (3, 2, 2, 1, 1),
            (3, 1, 2, 1, 3),
            (3, 1, 2, 1, 3),
        ],
    ),
]


@pytest.mark.parametrize("max_seq_len, max_patch_size, samples", testdata)
@pytest.mark.parametrize("observed_pct", [0, 0.25, 0.5, 0.75, 1])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_packed_nop_scaler(
    max_seq_len: int,
    max_patch_size: int,
    samples: list[tuple[int, int, int, int, int]],
    observed_pct: float,
    seed: int,
):
    def get_loc_scale_func(
        _target: torch.Tensor, _observed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dim = _target.shape[1]
        return torch.zeros(1, dim), torch.ones(1, dim)

    torch.manual_seed(seed)
    _test_packed_scaler(
        get_loc_scale_func,
        PackedNOPScaler(),
        samples,
        max_seq_len,
        max_patch_size,
        observed_pct,
    )


@pytest.mark.parametrize("max_seq_len, max_patch_size, samples", testdata)
@pytest.mark.parametrize("observed_pct", [0, 0.25, 0.5, 0.75, 1])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_packed_std_scaler(
    max_seq_len: int,
    max_patch_size: int,
    samples: list[tuple[int, int, int, int, int]],
    observed_pct: float,
    seed: int,
):
    def get_loc_scale_func(
        _target: torch.Tensor, _observed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _target = _target * _observed
        loc = safe_div(
            _target.sum(dim=(0, -1), keepdim=True),
            _observed.sum(dim=(0, -1), keepdim=True),
        )
        deviation = ((_target - loc) ** 2) * _observed
        var = safe_div(
            deviation.sum(dim=(0, -1), keepdim=True),
            torch.clamp(_observed.sum(dim=(0, -1), keepdim=True) - 1, min=0),
        )
        scale = torch.sqrt(var + 1e-5)
        return loc.squeeze(-1), scale.squeeze(-1)

    torch.manual_seed(seed)
    _test_packed_scaler(
        get_loc_scale_func,
        PackedStdScaler(),
        samples,
        max_seq_len,
        max_patch_size,
        observed_pct,
    )


@pytest.mark.parametrize("max_seq_len, max_patch_size, samples", testdata)
@pytest.mark.parametrize("observed_pct", [0, 0.25, 0.5, 0.75, 1])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_packed_abs_mean_scaler(
    max_seq_len: int,
    max_patch_size: int,
    samples: list[tuple[int, int, int, int, int]],
    observed_pct: float,
    seed: int,
):
    def get_loc_scale_func(
        _target: torch.Tensor, _observed: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dim = _target.shape[1]
        _target = _target * _observed
        # loc = safe_div(
        #     _target.sum(dim=(0, -1), keepdim=True),
        #     _observed.sum(dim=(0, -1), keepdim=True),
        # )
        # deviation = ((_target - loc) ** 2) * _observed
        # var = safe_div(
        #     deviation.sum(dim=(0, -1), keepdim=True),
        #     torch.clamp(_observed.sum(dim=(0, -1), keepdim=True) - 1, min=0),
        # )
        # scale = torch.sqrt(var + 1e-5)
        # return loc.squeeze(-1), scale.squeeze(-1)

        scale = safe_div(
            _target.abs().sum(dim=(0, -1), keepdim=True),
            _observed.sum(dim=(0, -1), keepdim=True),
        )

        return torch.zeros(1, dim), scale.squeeze(-1)

    torch.manual_seed(seed)
    _test_packed_scaler(
        get_loc_scale_func,
        PackedAbsMeanScaler(),
        samples,
        max_seq_len,
        max_patch_size,
        observed_pct,
    )
