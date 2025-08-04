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

# Normalized loss functions for time series forecasting


class PointNormType(Enum):
    """
    Enumeration of normalization types for point forecast losses.
    
    Normalization makes loss values more comparable across different scales and
    datasets by dividing the raw error by some characteristic of the target data.
    Different normalization strategies are appropriate for different use cases.
    """
    
    NONE = "none"  # No normalization, use raw errors
    ABS_TARGET = "absolute_target"  # Normalize by mean absolute target value for each observation
    ABS_TARGET_SQ = "absolute_target_squared"  # Matrix factorization definition of NRMSE/ND
    TARGET = "target"  # Normalize by mean target value for each observation
    TARGET_SQ = "target_squared"  # Classical definition of NRMSE/NMAE
    STD_DEV = (
        "standard_deviation"  # Normalize by standard deviation of target for each observation
    )
    VAR = "variance"  # Normalize by variance of target for each observation
    # MAX_MIN = "max_min"  # Normalize by range (max - min) of target values
    # IQR = "interquartile_range"  # Normalize by interquartile range of target values


@abstract_class_property("error_func")
class PackedPointNormalizedLoss(PackedPointLoss, abc.ABC):
    """
    Abstract base class for normalized point forecast loss functions.
    
    This class provides a framework for implementing loss functions that normalize
    the error by some characteristic of the target data. Subclasses must define
    an `error_func` property that computes the raw error between predictions and targets.
    
    The normalization is applied by dividing the raw error by a denominator computed
    based on the specified normalization type. This makes the loss values more
    comparable across different scales and datasets.
    """
    
    error_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = NotImplemented

    def __init__(
        self,
        normalize: PointNormType = PointNormType.NONE,
        correction: int = 1,
        epsilon: float = 1e-5,
    ):
        """
        Initialize the normalized loss function.
        
        Args:
            normalize: Type of normalization to apply. Default is NONE (no normalization).
            correction: Correction factor for variance calculation (Bessel's correction).
                       Default is 1 for unbiased estimation.
            epsilon: Small constant added to denominators to prevent division by zero.
                    Default is 1e-5.
        """
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
        """
        Compute the normalized loss between predictions and targets.
        
        This method computes the raw error using the subclass-defined error_func,
        then normalizes it by dividing by a denominator based on the specified
        normalization type.
        
        Args:
            pred: Predicted values.
            target: Target values.
            prediction_mask: Binary mask for predictions.
            observed_mask: Binary mask for observed values.
            sample_id: Sample identifiers.
            variate_id: Variate identifiers.
            
        Returns:
            Normalized loss per token.
        """
        loss = self.error_func(pred, target)
        denominator = self.denominator_func(
            target, observed_mask, sample_id, variate_id
        )
        loss = safe_div(loss, denominator)
        return loss

    @property
    def denominator_func(self) -> Callable:
        """
        Get the appropriate denominator function based on the normalization type.
        
        This property maps the normalization type to the corresponding denominator
        function that will be used to normalize the raw error.
        
        Returns:
            Function that computes the denominator for normalization.
        
        Raises:
            ValueError: If an invalid normalization type is specified.
        """
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
    """
    Normalized Mean Absolute Error (NMAE) loss for time series forecasting.
    
    NMAE normalizes the Mean Absolute Error (MAE) by a characteristic of the target data,
    such as the mean absolute value, mean value, or standard deviation. This makes the
    loss values more comparable across different scales and datasets.
    
    The normalization type is specified during initialization and determines how the
    raw MAE is normalized. For example, with ABS_TARGET normalization, the loss becomes
    mean(|actual - predicted|) / mean(|actual|), which is similar to MAPE but with a
    different denominator aggregation.
    """
    
    error_func = torch.nn.L1Loss(reduction="none")


class PackedNMSELoss(PackedPointNormalizedLoss):
    """
    Normalized Mean Squared Error (NMSE) loss for time series forecasting.
    
    NMSE normalizes the Mean Squared Error (MSE) by a characteristic of the target data,
    such as the squared mean value, variance, or squared mean absolute value. This makes
    the loss values more comparable across different scales and datasets.
    
    The normalization type is specified during initialization and determines how the
    raw MSE is normalized. For example, with VAR normalization, the loss becomes
    mean((actual - predicted)²) / var(actual), which measures the error relative to
    the inherent variability of the target data.
    """
    
    error_func = torch.nn.MSELoss(reduction="none")


class PackedNRMSELoss(PackedPointNormalizedLoss):
    """
    Normalized Root Mean Squared Error (NRMSE) loss for time series forecasting.
    
    NRMSE normalizes the Root Mean Squared Error (RMSE) by a characteristic of the target data,
    such as the mean value, standard deviation, or mean absolute value. This makes the
    loss values more comparable across different scales and datasets.
    
    The normalization type is specified during initialization and determines how the
    raw RMSE is normalized. For example, with STD_DEV normalization, the loss becomes
    sqrt(mean((actual - predicted)²)) / std(actual), which measures the error in units
    of standard deviations.
    
    This class overrides the reduce_loss method to apply the square root operation
    before normalization, which distinguishes it from NMSELoss.
    """
    
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
    """
    Normalized Mean Logarithmic Squared Error (NMLSE) loss for time series forecasting.
    
    NMLSE applies a logarithmic transformation to the normalized MSE, which can be useful
    for handling data with exponential growth or when errors span multiple orders of magnitude.
    It is calculated as: log(1 + MSE / variance).
    
    This loss is always normalized by the variance of the target data (PointNormType.VAR),
    as this provides a natural scale for the logarithmic transformation. The logarithmic
    transformation makes the loss less sensitive to large errors compared to NMSE.
    """
    
    error_func = torch.nn.MSELoss(reduction="none")

    def __init__(self, df: float = 1.0):
        """
        Initialize the NMLSE loss.
        
        Args:
            df: Degrees of freedom parameter that scales the normalized MSE before
                applying the logarithmic transformation. Default is 1.0.
        """
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
        """
        Compute the NMLSE loss between predictions and targets.
        
        This method first computes the normalized MSE using the parent class method,
        then applies a logarithmic transformation using log1p (log(1+x)) to reduce
        sensitivity to large errors.
        
        Args:
            pred: Predicted values.
            target: Target values.
            prediction_mask: Binary mask for predictions.
            observed_mask: Binary mask for observed values.
            sample_id: Sample identifiers.
            variate_id: Variate identifiers.
            
        Returns:
            NMLSE loss per token.
        """
        loss = super()._loss_func(
            pred, target, prediction_mask, observed_mask, sample_id, variate_id
        )
        return torch.log1p(loss / self.df)
