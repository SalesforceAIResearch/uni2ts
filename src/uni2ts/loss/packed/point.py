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

from .normalized import PackedNMAELoss, PackedNMSELoss, PackedNRMSELoss, PointNormType

# Basic point forecast loss functions without normalization


class PackedMAELoss(PackedNMAELoss):
    """
    Mean Absolute Error (MAE) loss for time series forecasting.
    
    MAE measures the average absolute difference between predicted and actual values.
    It is calculated as: mean(|actual - predicted|)
    
    This loss treats all errors equally regardless of their magnitude, making it less
    sensitive to outliers compared to MSE. It is implemented as a special case of
    the normalized MAE loss with no normalization.
    """
    
    def __init__(self):
        """
        Initialize the MAE loss with no normalization.
        """
        super().__init__(normalize=PointNormType.NONE)


class PackedMSELoss(PackedNMSELoss):
    """
    Mean Squared Error (MSE) loss for time series forecasting.
    
    MSE measures the average squared difference between predicted and actual values.
    It is calculated as: mean((actual - predicted)²)
    
    This loss penalizes larger errors more heavily than smaller ones due to the
    squaring operation. It is implemented as a special case of the normalized MSE
    loss with no normalization.
    """
    
    def __init__(self):
        """
        Initialize the MSE loss with no normalization.
        """
        super().__init__(normalize=PointNormType.NONE)


class PackedRMSELoss(PackedNRMSELoss):
    """
    Root Mean Squared Error (RMSE) loss for time series forecasting.
    
    RMSE is the square root of the MSE, providing a measure of error in the same
    units as the target variable. It is calculated as: sqrt(mean((actual - predicted)²))
    
    Like MSE, RMSE penalizes larger errors more heavily, but the square root operation
    brings the scale back to that of the original data. It is implemented as a special
    case of the normalized RMSE loss with no normalization.
    """
    
    def __init__(self):
        """
        Initialize the RMSE loss with no normalization.
        """
        super().__init__(normalize=PointNormType.NONE)
