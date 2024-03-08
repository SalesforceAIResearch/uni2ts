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
import torch.nn.functional as F
from einops import rearrange, repeat
from jaxtyping import Bool, Float

from uni2ts.common.torch_util import safe_div
from uni2ts.loss.packed import (
    PackedDistributionLoss,
    PackedLoss,
    PackedMAELoss,
    PackedMAPELoss,
    PackedNLLLoss,
    PackedNMAELoss,
    PackedNMLSELoss,
    PackedNRMSELoss,
    PackedPointLoss,
    PackedPointNormalizedLoss,
    PackedRMSELoss,
    PackedSMAPELoss,
    PointNormType,
)


def pack_seq(
    xs: list[torch.Tensor],
    max_seq_len: int,
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
                torch.zeros(max_seq_len - curr_len, *shape, dtype=batch[i][0].dtype)
            )

    return torch.stack([torch.cat(x, dim=0) for x in batch], dim=0)


def _test_packed_loss(
    loss_accum_func: Callable[
        [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
    ],
    packed_loss_func: PackedLoss,
    samples: list[tuple[int, int, int, int, int]],
    max_seq_len: int,
    max_patch_size: int,
    observed_pct: float,
    possible_nan: bool = False,
):
    data = {
        "inp": [],
        "target": [],
        "prediction_mask": [],
        "sample_id": [],
        "observed_mask": [],
        "variate_id": [],
    }
    sample_id_counter = 1
    packing_counter = 0
    loss = torch.zeros(())
    loss_counter = 0
    for time, n_target_time, target_dim, cov_dim, patch_size in samples:
        dim = target_dim + cov_dim
        assert time * dim <= max_seq_len

        inp = torch.randn(time, dim, max_patch_size)
        target = torch.randn(time, dim, max_patch_size)
        observed_mask = (
            torch.zeros(time, dim, max_patch_size, dtype=torch.bool) < observed_pct
        )
        prediction_mask = torch.zeros(time, dim, dtype=torch.bool)
        prediction_mask[-n_target_time:, -target_dim:] = True

        loss += loss_accum_func(inp, target, observed_mask, prediction_mask)
        loss_counter += target_dim

        inp = rearrange(inp, "t d p -> (d t) p")
        target = rearrange(target, "t d p -> (d t) p")
        prediction_mask = rearrange(prediction_mask, "t d -> (d t)")
        observed_mask = rearrange(observed_mask, "t d p -> (d t) p")
        variate_id = repeat(torch.arange(dim), "d -> (d t)", t=time)
        sample_id = torch.ones(time * dim, dtype=torch.long) * sample_id_counter

        if packing_counter + time * dim > max_seq_len:
            sample_id_counter = 1
            packing_counter = 0
        else:
            sample_id_counter += 1
            packing_counter += time * dim

        data["inp"].append(inp)
        data["target"].append(target)
        data["prediction_mask"].append(prediction_mask)
        data["sample_id"].append(sample_id)
        data["observed_mask"].append(observed_mask)
        data["variate_id"].append(variate_id)

    loss = safe_div(loss, torch.as_tensor(loss_counter))

    inp = pack_seq(data["inp"], max_seq_len)
    target = pack_seq(data["target"], max_seq_len)
    prediction_mask = pack_seq(data["prediction_mask"], max_seq_len)
    sample_id = pack_seq(data["sample_id"], max_seq_len)
    observed_mask = pack_seq(data["observed_mask"], max_seq_len)
    variate_id = pack_seq(data["variate_id"], max_seq_len)

    packed_loss = packed_loss_func(
        pred=inp,
        target=target,
        prediction_mask=prediction_mask,
        observed_mask=observed_mask,
        sample_id=sample_id,
        variate_id=variate_id,
    )
    assert packed_loss.shape == ()

    if possible_nan and torch.isnan(packed_loss):
        assert torch.isnan(loss)
    else:
        assert torch.isclose(packed_loss, loss, atol=1e-5)
        assert not torch.isnan(packed_loss)


testdata = [
    (
        5,  # max_seq_len
        2,  # max_patch_size
        [
            (3, 1, 1, 0, 2),
            (2, 1, 1, 0, 1),
        ],  # time, n_target_time, target_dim, cov_dim, patch_size
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
def test_packed_rmse_loss(
    max_seq_len: int,
    max_patch_size: int,
    samples: list[tuple[int, int, int, int, int]],
    observed_pct: float,
    seed: int,
):
    def rmse_loss_accum(
        inp: Float[torch.Tensor, "time dim patch"],
        target: Float[torch.Tensor, "time dim patch"],
        observed: Bool[torch.Tensor, "time dim patch"],
        prediction: Bool[torch.Tensor, "time dim"],
    ) -> Float[torch.Tensor, ""]:
        inp = inp * observed
        target = target * observed
        loss = F.mse_loss(inp, target, reduction="none")
        out = torch.zeros(())
        for d in range(inp.shape[1]):
            mask = observed[:, d] * prediction[:, d].unsqueeze(-1)
            if mask.sum() > 0:
                out += torch.sqrt(loss[:, d][mask].mean())
        return out

    torch.manual_seed(seed)
    _test_packed_loss(
        rmse_loss_accum,
        PackedRMSELoss(),
        samples,
        max_seq_len,
        max_patch_size,
        observed_pct,
    )


@pytest.mark.parametrize("max_seq_len, max_patch_size, samples", testdata)
@pytest.mark.parametrize("observed_pct", [0, 0.25, 0.5, 0.75, 1])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_packed_mae_loss(
    max_seq_len: int,
    max_patch_size: int,
    samples: list[tuple[int, int, int, int, int]],
    observed_pct: float,
    seed: int,
):
    def mae_loss_accum(
        inp: Float[torch.Tensor, "time dim patch"],
        target: Float[torch.Tensor, "time dim patch"],
        observed: Bool[torch.Tensor, "time dim patch"],
        prediction: Bool[torch.Tensor, "time dim"],
    ) -> Float[torch.Tensor, ""]:
        inp = inp * observed
        target = target * observed
        loss = F.l1_loss(inp, target, reduction="none")
        loss = safe_div(
            (loss * prediction.unsqueeze(-1)).sum((0, 2)),
            (observed * prediction.unsqueeze(-1)).sum((0, 2)),
        )
        return loss.sum()

    torch.manual_seed(seed)
    _test_packed_loss(
        mae_loss_accum,
        PackedMAELoss(),
        samples,
        max_seq_len,
        max_patch_size,
        observed_pct,
    )


@pytest.mark.parametrize("max_seq_len, max_patch_size, samples", testdata)
@pytest.mark.parametrize("observed_pct", [0, 0.25, 0.5, 0.75, 1])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_packed_mape_loss(
    max_seq_len: int,
    max_patch_size: int,
    samples: list[tuple[int, int, int, int, int]],
    observed_pct: float,
    seed: int,
):
    def mape_loss_accum(
        inp: Float[torch.Tensor, "time dim patch"],
        target: Float[torch.Tensor, "time dim patch"],
        observed: Bool[torch.Tensor, "time dim patch"],
        prediction: Bool[torch.Tensor, "time dim"],
    ) -> Float[torch.Tensor, ""]:
        inp = inp * observed
        target = target * observed
        loss = F.l1_loss(inp, target, reduction="none")
        loss = 100 * safe_div(loss, target.abs())
        loss = safe_div(
            (loss * prediction.unsqueeze(-1)).sum((0, 2)),
            (observed * prediction.unsqueeze(-1)).sum((0, 2)),
        )
        return loss.sum()

    torch.manual_seed(seed)
    _test_packed_loss(
        mape_loss_accum,
        PackedMAPELoss(),
        samples,
        max_seq_len,
        max_patch_size,
        observed_pct,
    )


@pytest.mark.parametrize("max_seq_len, max_patch_size, samples", testdata)
@pytest.mark.parametrize("observed_pct", [0, 0.25, 0.5, 0.75, 1])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_packed_smape_loss(
    max_seq_len: int,
    max_patch_size: int,
    samples: list[tuple[int, int, int, int, int]],
    observed_pct: float,
    seed: int,
):
    def smape_loss_accum(
        inp: Float[torch.Tensor, "time dim patch"],
        target: Float[torch.Tensor, "time dim patch"],
        observed: Bool[torch.Tensor, "time dim patch"],
        prediction: Bool[torch.Tensor, "time dim"],
    ) -> Float[torch.Tensor, ""]:
        inp = inp * observed
        target = target * observed
        loss = F.l1_loss(inp, target, reduction="none")
        loss = 200 * safe_div(loss, (inp.abs() + target.abs()))
        loss = safe_div(
            (loss * prediction.unsqueeze(-1)).sum((0, 2)),
            (observed * prediction.unsqueeze(-1)).sum((0, 2)),
        )
        return loss.sum()

    torch.manual_seed(seed)
    _test_packed_loss(
        smape_loss_accum,
        PackedSMAPELoss(),
        samples,
        max_seq_len,
        max_patch_size,
        observed_pct,
    )


@pytest.mark.parametrize("max_seq_len, max_patch_size, samples", testdata)
@pytest.mark.parametrize("observed_pct", [0, 0.25, 0.5, 0.75, 1])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("normalize", [norm_type for norm_type in PointNormType])
def test_packed_nmae_loss(
    max_seq_len: int,
    max_patch_size: int,
    samples: list[tuple[int, int, int, int, int]],
    observed_pct: float,
    seed: int,
    normalize: PointNormType,
):
    def nmae_loss_accum(
        inp: Float[torch.Tensor, "time dim patch"],
        target: Float[torch.Tensor, "time dim patch"],
        observed: Bool[torch.Tensor, "time dim patch"],
        prediction: Bool[torch.Tensor, "time dim"],
    ) -> Float[torch.Tensor, ""]:
        inp = inp * observed
        target = target * observed
        loss = F.l1_loss(inp, target, reduction="none")

        if normalize == PointNormType.NONE:
            ...
        elif normalize == PointNormType.ABS_TARGET:
            denom = safe_div(
                target.abs().sum(dim=(0, -1), keepdim=True),
                observed.sum(dim=(0, -1), keepdim=True),
            )
            loss = safe_div(loss, denom)
        elif normalize == PointNormType.ABS_TARGET_SQ:
            denom = safe_div(
                target.abs().sum(dim=(0, -1), keepdim=True),
                observed.sum(dim=(0, -1), keepdim=True),
            ).pow(2)
            loss = safe_div(loss, denom)
        elif normalize == PointNormType.TARGET:
            denom = safe_div(
                target.sum(dim=(0, -1), keepdim=True),
                observed.sum(dim=(0, -1), keepdim=True),
            )
            loss = safe_div(loss, denom)
        elif normalize == PointNormType.TARGET_SQ:
            denom = safe_div(
                target.sum(dim=(0, -1), keepdim=True),
                observed.sum(dim=(0, -1), keepdim=True),
            ).pow(2)
            loss = safe_div(loss, denom)
        elif normalize == PointNormType.STD_DEV:
            loc = safe_div(
                target.sum(dim=(0, -1), keepdim=True),
                observed.sum(dim=(0, -1), keepdim=True),
            )
            var = safe_div(
                (target - loc).pow(2).sum(dim=(0, -1), keepdim=True),
                torch.clamp(observed.sum(dim=(0, -1), keepdim=True) - 1, 0),
            )
            denom = torch.sqrt(var + 1e-5)
            loss = safe_div(loss, denom)
        elif normalize == PointNormType.VAR:
            loc = safe_div(
                target.sum(dim=(0, -1), keepdim=True),
                observed.sum(dim=(0, -1), keepdim=True),
            )
            denom = safe_div(
                (target - loc).pow(2).sum(dim=(0, -1), keepdim=True),
                torch.clamp(observed.sum(dim=(0, -1), keepdim=True) - 1, 0),
            )
            loss = safe_div(loss, denom)

        loss = safe_div(
            (loss * prediction.unsqueeze(-1)).sum((0, 2)),
            (observed * prediction.unsqueeze(-1)).sum((0, 2)),
        )
        return loss.sum()

    torch.manual_seed(seed)
    _test_packed_loss(
        nmae_loss_accum,
        PackedNMAELoss(normalize=normalize),
        samples,
        max_seq_len,
        max_patch_size,
        observed_pct,
    )


@pytest.mark.parametrize("max_seq_len, max_patch_size, samples", testdata)
@pytest.mark.parametrize("observed_pct", [0, 0.25, 0.5, 0.75, 1])
@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("normalize", [norm_type for norm_type in PointNormType])
def test_packed_nrmse_loss(
    max_seq_len: int,
    max_patch_size: int,
    samples: list[tuple[int, int, int, int, int]],
    observed_pct: float,
    seed: int,
    normalize: PointNormType,
):
    def nrmse_loss_accum(
        inp: Float[torch.Tensor, "time dim patch"],
        target: Float[torch.Tensor, "time dim patch"],
        observed: Bool[torch.Tensor, "time dim patch"],
        prediction: Bool[torch.Tensor, "time dim"],
    ) -> Float[torch.Tensor, ""]:
        inp = inp * observed
        target = target * observed
        loss = F.mse_loss(inp, target, reduction="none")
        if normalize == PointNormType.NONE:
            ...
        elif normalize == PointNormType.ABS_TARGET:
            denom = safe_div(
                target.abs().sum(dim=(0, -1), keepdim=True),
                observed.sum(dim=(0, -1), keepdim=True),
            )
            loss = safe_div(loss, denom)
        elif normalize == PointNormType.ABS_TARGET_SQ:
            denom = safe_div(
                target.abs().sum(dim=(0, -1), keepdim=True),
                observed.sum(dim=(0, -1), keepdim=True),
            ).pow(2)
            loss = safe_div(loss, denom)
        elif normalize == PointNormType.TARGET:
            denom = safe_div(
                target.sum(dim=(0, -1), keepdim=True),
                observed.sum(dim=(0, -1), keepdim=True),
            )
            loss = safe_div(loss, denom)
        elif normalize == PointNormType.TARGET_SQ:
            denom = safe_div(
                target.sum(dim=(0, -1), keepdim=True),
                observed.sum(dim=(0, -1), keepdim=True),
            ).pow(2)
            loss = safe_div(loss, denom)
        elif normalize == PointNormType.STD_DEV:
            loc = safe_div(
                target.sum(dim=(0, -1), keepdim=True),
                observed.sum(dim=(0, -1), keepdim=True),
            )
            var = safe_div(
                (target - loc).pow(2).sum(dim=(0, -1), keepdim=True),
                torch.clamp(observed.sum(dim=(0, -1), keepdim=True) - 1, 0),
            )
            denom = torch.sqrt(var + 1e-5)
            loss = safe_div(loss, denom)
        elif normalize == PointNormType.VAR:
            loc = safe_div(
                target.sum(dim=(0, -1), keepdim=True),
                observed.sum(dim=(0, -1), keepdim=True),
            )
            denom = safe_div(
                (target - loc).pow(2).sum(dim=(0, -1), keepdim=True),
                torch.clamp(observed.sum(dim=(0, -1), keepdim=True) - 1, 0),
            )
            loss = safe_div(loss, denom)

        out = torch.zeros(())
        for d in range(inp.shape[1]):
            mask = observed[:, d] * prediction[:, d].unsqueeze(-1)
            if mask.sum() > 0:
                out += torch.sqrt(loss[:, d][mask].mean())
        return out

    torch.manual_seed(seed)
    _test_packed_loss(
        nrmse_loss_accum,
        PackedNRMSELoss(normalize=normalize),
        samples,
        max_seq_len,
        max_patch_size,
        observed_pct,
        possible_nan=normalize == PointNormType.TARGET,
    )


@pytest.mark.parametrize("max_seq_len, max_patch_size, samples", testdata)
@pytest.mark.parametrize("observed_pct", [0, 0.25, 0.5, 0.75, 1])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_packed_nmlse_loss(
    max_seq_len: int,
    max_patch_size: int,
    samples: list[tuple[int, int, int, int, int]],
    observed_pct: float,
    seed: int,
):
    def nmlse_loss_accum(
        inp: Float[torch.Tensor, "time dim patch"],
        target: Float[torch.Tensor, "time dim patch"],
        observed: Bool[torch.Tensor, "time dim patch"],
        prediction: Bool[torch.Tensor, "time dim"],
    ) -> Float[torch.Tensor, ""]:
        inp = inp * observed
        target = target * observed
        loss = F.mse_loss(inp, target, reduction="none")
        loc = safe_div(
            target.sum(dim=(0, -1), keepdim=True),
            observed.sum(dim=(0, -1), keepdim=True),
        )
        denom = safe_div(
            (target - loc).pow(2).sum(dim=(0, -1), keepdim=True),
            torch.clamp(observed.sum(dim=(0, -1), keepdim=True) - 1, 0),
        )
        loss = safe_div(loss, denom)
        loss = torch.log(1 + loss)
        loss = safe_div(
            (loss * prediction.unsqueeze(-1)).sum((0, 2)),
            (observed * prediction.unsqueeze(-1)).sum((0, 2)),
        )
        return loss.sum()

    torch.manual_seed(seed)
    _test_packed_loss(
        nmlse_loss_accum,
        PackedNMLSELoss(),
        samples,
        max_seq_len,
        max_patch_size,
        observed_pct,
    )
