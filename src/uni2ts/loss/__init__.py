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
Loss functions for time series forecasting models.

This module provides various loss functions for training time series forecasting models,
with a focus on handling packed sequences (variable-length time series). The loss functions
are organized into different categories:

- Point losses: For evaluating point forecasts (e.g., MAE, MSE, RMSE)
- Distribution losses: For evaluating probabilistic forecasts (e.g., NLL)
- Normalized losses: For evaluating forecasts with normalization (e.g., NMAE, NRMSE)
- Percentage error losses: For evaluating forecasts with percentage errors (e.g., MAPE, SMAPE)

All loss functions support masking for handling missing values and variable-length sequences.
"""
