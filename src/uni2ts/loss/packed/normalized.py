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
from enum import Enum
from typing import Callable, Optional

import torch
from einops import reduce
from jaxtyping import Bool, Float, Int

from uni2ts.common.core import abstract_class_property
from uni2ts.common.torch_util import safe_div

from ._base import PackedPointLoss


class PointNormType(Enum):
    NONE = "none"
    ABS_TARGET = "absolute_target"  # normalize by mean abs_target for each obs
    ABS_TARGET_SQ = "absolute_target_squared"  # matfact def of NRMSE/ND
    TARGET = "target"  # normalize by mean target for each obs
    TARGET_SQ = "target_squared"  # classical def of NRMSE/NMAE
    STD_DEV = (
        "standard_deviation"  # normalize by standard deviation of target for each obs
    )
    VAR = "variance"  # normalize by variance of target for each obs
    # MAX_MIN = "max_min"
    # IQR = "interquartile_range"


@abstract_class_property("error_func")
class PackedPointNormalizedLoss(PackedPointLoss, abc.ABC):
    error_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = NotImplemented

    def __init__(
        self,
        normalize: PointNormType = PointNormType.NONE,
        correction: int = 1,
        epsilon: float = 1e-5,
    ):
        super().__init__()
        self.normalize = PointNormType(normalize)
        self.correction = correction
        self.epsilon = epsilon

    def _loss_func(
        self,
        pred: Float[torch.Tensor, "*batch seq_len #dim"],
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]:
        loss = self.error_func(pred, target)
        denominator = self.denominator_func(
            target, observed_mask, sample_id, variate_id
        )
        loss = safe_div(loss, denominator)
        return loss

    @property
    def denominator_func(self) -> Callable:
        func_map = {
            PointNormType.NONE: self.none_denominator,
            PointNormType.ABS_TARGET: self.abs_target_denominator,
            PointNormType.ABS_TARGET_SQ: self.abs_target_sq_denominator,
            PointNormType.TARGET: self.target_denominator,
            PointNormType.TARGET_SQ: self.target_sq_denominator,
            PointNormType.STD_DEV: self.std_dev_denominator,
            PointNormType.VAR: self.var_denominator,
        }
        if self.normalize not in func_map:
            raise ValueError(f"Invalid normalize type '{self.normalize}'")
        return func_map[self.normalize]

    @staticmethod
    def none_denominator(
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]:
        return torch.ones_like(target)

    @staticmethod
    def reduce_denominator(
        value: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]:
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        tobs = reduce(
            id_mask * reduce(observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        value = reduce(
            id_mask * reduce(value * observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        value = safe_div(value, tobs)
        return value

    def abs_target_denominator(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]:
        return self.reduce_denominator(
            target.abs(), observed_mask, sample_id, variate_id
        )

    def abs_target_sq_denominator(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]:
        return torch.pow(
            self.reduce_denominator(target.abs(), observed_mask, sample_id, variate_id),
            2,
        )

    def target_denominator(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]:
        return self.reduce_denominator(target, observed_mask, sample_id, variate_id)

    def target_sq_denominator(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]:
        return torch.pow(
            self.reduce_denominator(target, observed_mask, sample_id, variate_id), 2
        )

    def std_dev_denominator(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]:
        var = self.var_denominator(target, observed_mask, sample_id, variate_id)
        std_dev = torch.sqrt(var + self.epsilon)
        return std_dev

    def var_denominator(
        self,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]:
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        tobs = reduce(
            id_mask * reduce(observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loc = reduce(
            id_mask * reduce(target * observed_mask, "... seq dim -> ... 1 seq", "sum"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loc = safe_div(loc, tobs)
        var = reduce(
            id_mask
            * reduce(
                ((target - loc) ** 2) * observed_mask, "... seq dim -> ... 1 seq", "sum"
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        var = safe_div(var, (tobs - self.correction))
        return var


class PackedNMAELoss(PackedPointNormalizedLoss):
    error_func = torch.nn.L1Loss(reduction="none")


class PackedNMSELoss(PackedPointNormalizedLoss):
    error_func = torch.nn.MSELoss(reduction="none")


class PackedNRMSELoss(PackedPointNormalizedLoss):
    error_func = torch.nn.MSELoss(reduction="none")

    def reduce_loss(
        self,
        loss: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Optional[Bool[torch.Tensor, "*batch seq_len"]],
        observed_mask: Optional[Bool[torch.Tensor, "*batch seq_len #dim"]],
        sample_id: Optional[Int[torch.Tensor, "*batch seq_len"]],
        variate_id: Optional[Int[torch.Tensor, "*batch seq_len"]],
    ) -> Float[torch.Tensor, ""]:
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        mask = prediction_mask.unsqueeze(-1) * observed_mask
        loss = reduce(
            id_mask
            * reduce(
                loss * mask,
                "... seq dim -> ... 1 seq",
                "sum",
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loss = torch.sqrt(loss + self.epsilon)
        tobs = reduce(
            id_mask
            * reduce(
                mask,
                "... seq dim -> ... 1 seq",
                "sum",
            ),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        )
        loss = safe_div(loss, torch.sqrt(tobs))

        return super().reduce_loss(
            loss, prediction_mask, observed_mask, sample_id, variate_id
        )


class PackedNMLSELoss(PackedPointNormalizedLoss):
    error_func = torch.nn.MSELoss(reduction="none")

    def __init__(self, df: float = 1.0):
        super().__init__(PointNormType.VAR)
        self.df = df

    def _loss_func(
        self,
        pred: Float[torch.Tensor, "*batch seq_len #dim"],
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]:
        loss = super()._loss_func(
            pred, target, prediction_mask, observed_mask, sample_id, variate_id
        )
        return torch.log1p(loss / self.df)
