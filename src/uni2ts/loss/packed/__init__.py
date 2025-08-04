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

"""
Packed loss functions for time series forecasting.

This submodule provides loss functions that support packed sequences, which are
variable-length time series batched together efficiently. These loss functions
handle masking for predictions and observations, and support sample and variate
identifiers for proper aggregation across batches.

The loss functions are organized into several categories:
- Base classes: Abstract base classes for different types of losses
- Distribution losses: For probabilistic forecasts (distributions)
- Normalized losses: With various normalization strategies
- Percentage error losses: For relative error metrics
- Point losses: For deterministic forecasts
"""

from ._base import PackedDistributionLoss, PackedLoss, PackedPointLoss
from .distribution import PackedNLLLoss
from .normalized import (
    PackedNMAELoss,
    PackedNMLSELoss,
    PackedNMSELoss,
    PackedNRMSELoss,
    PackedPointNormalizedLoss,
    PointNormType,
)
from .percentage_error import PackedMAPELoss, PackedSMAPELoss
from .point import PackedMAELoss, PackedMSELoss, PackedRMSELoss

__all__ = [
    "PackedDistributionLoss",
    "PackedLoss",
    "PackedMAELoss",
    "PackedMAPELoss",
    "PackedMSELoss",
    "PackedNLLLoss",
    "PackedNMAELoss",
    "PackedNMLSELoss",
    "PackedNMSELoss",
    "PackedNRMSELoss",
    "PackedPointLoss",
    "PackedPointNormalizedLoss",
    "PackedRMSELoss",
    "PackedSMAPELoss",
    "PointNormType",
]
