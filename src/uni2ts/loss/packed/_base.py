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
from typing import Any, Optional

import torch
from einops import rearrange, reduce
from jaxtyping import Bool, Float, Int
from torch.distributions import Distribution

from uni2ts.common.torch_util import safe_div

# Base classes for packed loss functions


class PackedLoss(abc.ABC):
    """
    Abstract base class for loss functions supporting packed inputs.
    Subclasses should implement the _loss_func method which computes the loss function per token.
    
    This is the foundation for all loss functions in the packed module. It handles the common
    operations of applying masks, dealing with sample and variate identifiers, and reducing
    the loss appropriately across the batch.
    """

    def __call__(
        self,
        pred: Any,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Optional[Bool[torch.Tensor, "*batch seq_len"]],
        observed_mask: Optional[Bool[torch.Tensor, "*batch seq_len #dim"]] = None,
        sample_id: Optional[Int[torch.Tensor, "*batch seq_len"]] = None,
        variate_id: Optional[Int[torch.Tensor, "*batch seq_len"]] = None,
    ) -> Float[torch.Tensor, ""]:
        """
        Computes the loss between predictions and targets.
        
        Args:
            pred: Predictions from the model. The type depends on the specific loss function.
                 For point forecasts, this is a tensor. For distribution forecasts, this is a
                 Distribution object.
            target: Target values (ground truth) with shape [*batch, seq_len, #dim].
            prediction_mask: Binary mask indicating which positions contain predictions (1) vs.
                            non-predictions (0). Shape: [*batch, seq_len].
            observed_mask: Binary mask indicating which values in the target are observed (1) vs.
                          missing (0). Shape: [*batch, seq_len, #dim]. If None, all values are
                          assumed to be observed.
            sample_id: Integer tensor identifying which sample each position belongs to.
                      Shape: [*batch, seq_len]. Used for proper aggregation of losses across
                      variable-length sequences. If None, all positions are assumed to be from
                      the same sample.
            variate_id: Integer tensor identifying which variate (time series) each position
                       belongs to. Shape: [*batch, seq_len]. Used for proper aggregation of
                       losses across multiple time series. If None, all positions are assumed
                       to be from the same variate.
                       
        Returns:
            Scalar loss value.
        """
        if observed_mask is None:
            observed_mask = torch.ones_like(target, dtype=torch.bool)
        if sample_id is None:
            sample_id = torch.zeros_like(prediction_mask, dtype=torch.long)
        if variate_id is None:
            variate_id = torch.zeros_like(prediction_mask, dtype=torch.long)

        loss = self._loss_func(
            pred, target, prediction_mask, observed_mask, sample_id, variate_id
        )
        return self.reduce_loss(
            loss, prediction_mask, observed_mask, sample_id, variate_id
        )

    @abc.abstractmethod
    def _loss_func(
        self,
        pred: Any,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]: ...

    def reduce_loss(
        self,
        loss: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Optional[Bool[torch.Tensor, "*batch seq_len"]],
        observed_mask: Optional[Bool[torch.Tensor, "*batch seq_len #dim"]],
        sample_id: Optional[Int[torch.Tensor, "*batch seq_len"]],
        variate_id: Optional[Int[torch.Tensor, "*batch seq_len"]],
    ) -> Float[torch.Tensor, ""]:
        """
        Reduces the per-token loss to a scalar value, accounting for masks and identifiers.
        
        This method handles the proper aggregation of losses across variable-length sequences
        and multiple time series. It creates an ID mask to identify positions belonging to the
        same sample and variate, counts the number of observations per sample/variate, and
        normalizes the loss accordingly.
        
        Args:
            loss: Per-token loss with shape [*batch, seq_len, #dim].
            prediction_mask: Binary mask for predictions.
            observed_mask: Binary mask for observed values.
            sample_id: Sample identifiers.
            variate_id: Variate identifiers.
            
        Returns:
            Scalar loss value.
        """
        id_mask = torch.logical_and(
            torch.eq(sample_id.unsqueeze(-1), sample_id.unsqueeze(-2)),
            torch.eq(variate_id.unsqueeze(-1), variate_id.unsqueeze(-2)),
        )
        mask = prediction_mask.unsqueeze(-1) * observed_mask
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
        nobs = reduce(
            id_mask * rearrange(prediction_mask, "... seq -> ... 1 seq"),
            "... seq1 seq2 -> ... seq1 1",
            "sum",
        ) * prediction_mask.unsqueeze(-1)
        nobs = torch.where(nobs == 0, nobs, 1 / nobs).sum()
        loss = safe_div(loss, tobs * nobs)
        return (loss * mask).sum()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class PackedPointLoss(PackedLoss):
    """
    Abstract base class for loss functions on point forecasts.
    
    Point forecasts are deterministic predictions of future values (as opposed to
    probabilistic forecasts which predict distributions). This class serves as the
    base for loss functions like MAE, MSE, RMSE, etc.
    """

    @abc.abstractmethod
    def _loss_func(
        self,
        pred: Float[torch.Tensor, "*batch seq_len #dim"],
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]: ...


class PackedDistributionLoss(PackedLoss):
    """
    Abstract base class for loss functions on probabilistic (distribution) forecasts.
    
    Distribution forecasts predict probability distributions over future values rather
    than single point estimates. This class serves as the base for loss functions like
    negative log-likelihood (NLL) that evaluate probabilistic forecasts.
    """

    @abc.abstractmethod
    def _loss_func(
        self,
        pred: Distribution,
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]: ...
