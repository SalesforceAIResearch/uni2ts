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

import torch
from jaxtyping import Bool, Float, Int
from torch.nn import functional as F

from uni2ts.common.torch_util import safe_div

from ._base import PackedPointLoss

# Percentage error loss functions for time series forecasting


class PackedMAPELoss(PackedPointLoss):
    """
    Mean Absolute Percentage Error (MAPE) loss for time series forecasting.
    
    MAPE measures the average percentage difference between predicted and actual values.
    It is calculated as: 100% * mean(|actual - predicted| / |actual|)
    
    This loss is scale-independent, making it useful for comparing forecasts across
    different scales. However, it can be problematic when actual values are close to zero,
    as the percentage error can become very large or undefined.
    """
    
    def _loss_func(
        self,
        pred: Float[torch.Tensor, "*batch seq_len #dim"],
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]:
        """
        Computes the MAPE loss between predictions and targets.
        
        Args:
            pred: Predicted values.
            target: Target values.
            prediction_mask: Binary mask for predictions.
            observed_mask: Binary mask for observed values.
            sample_id: Sample identifiers.
            variate_id: Variate identifiers.
            
        Returns:
            MAPE loss per token (as a percentage).
        """
        loss = F.l1_loss(pred, target, reduction="none")
        loss = safe_div(loss, target.abs())
        return 100 * loss


class PackedSMAPELoss(PackedPointLoss):
    """
    Symmetric Mean Absolute Percentage Error (SMAPE) loss for time series forecasting.
    
    SMAPE is a variation of MAPE that is symmetric, meaning it treats over-forecasting and
    under-forecasting equally. It is calculated as:
    200% * mean(|actual - predicted| / (|actual| + |predicted|))
    
    This loss addresses some of the issues with MAPE, particularly when actual values are
    close to zero. The denominator (|actual| + |predicted|) is less likely to be zero
    compared to just |actual| in MAPE.
    """
    
    def _loss_func(
        self,
        pred: Float[torch.Tensor, "*batch seq_len #dim"],
        target: Float[torch.Tensor, "*batch seq_len #dim"],
        prediction_mask: Bool[torch.Tensor, "*batch seq_len"],
        observed_mask: Bool[torch.Tensor, "*batch seq_len #dim"],
        sample_id: Int[torch.Tensor, "*batch seq_len"],
        variate_id: Int[torch.Tensor, "*batch seq_len"],
    ) -> Float[torch.Tensor, "*batch seq_len #dim"]:
        """
        Computes the SMAPE loss between predictions and targets.
        
        Args:
            pred: Predicted values.
            target: Target values.
            prediction_mask: Binary mask for predictions.
            observed_mask: Binary mask for observed values.
            sample_id: Sample identifiers.
            variate_id: Variate identifiers.
            
        Returns:
            SMAPE loss per token (as a percentage).
        """
        loss = F.l1_loss(pred, target, reduction="none")
        loss = safe_div(loss, target.abs() + pred.detach().abs())
        return 200 * loss
