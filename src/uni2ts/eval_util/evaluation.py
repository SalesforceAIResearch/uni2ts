# Copyright 2018 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# A copy of the License is located at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# or in the "license" file accompanying this file. This file is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either
# express or implied. See the License for the specific language governing
# permissions and limitations under the License.

"""
Evaluation utilities for time series forecasting models.

This module provides functions and classes for evaluating forecasting models
by comparing their predictions against ground truth data. It supports various
evaluation metrics and aggregation methods to assess model performance.
"""

import logging
from collections import ChainMap
from dataclasses import dataclass
from typing import Iterable, List, Optional, Union

import numpy as np
import pandas as pd
from gluonts.dataset import DataEntry
from gluonts.dataset.split import TestData
from gluonts.ev.ts_stats import seasonal_error
from gluonts.itertools import batcher, prod
from gluonts.model import Forecast, Predictor
from gluonts.time_feature import get_seasonality
from toolz import first, valmap
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class BatchForecast:
    """
    Wrapper around ``Forecast`` objects, that adds a batch dimension
    to arrays returned by ``__getitem__``, for compatibility with
    ``gluonts.ev``.
    
    This class enables batch processing of multiple forecasts by stacking
    their outputs along a new dimension, making them compatible with the
    evaluation metrics in gluonts.ev.
    
    Attributes:
        forecasts: List of Forecast objects to be batched together
        allow_nan: Whether to allow NaN values in forecasts (default: False)
    """

    forecasts: List[Forecast]
    allow_nan: bool = False

    def __getitem__(self, name):
        """
        Retrieves a specific forecast attribute from all forecasts and stacks them.
        
        Args:
            name: The forecast attribute to retrieve (e.g., 'mean', 'samples')
            
        Returns:
            A numpy array with shape (batch_size, ...) containing the stacked attributes
            
        Raises:
            ValueError: If forecasts contain NaN values and allow_nan is False
        """
        values = [forecast[name].T for forecast in self.forecasts]
        res = np.stack(values, axis=0)

        if np.isnan(res).any():
            if not self.allow_nan:
                raise ValueError("Forecast contains NaN values")

            logger.warning("Forecast contains NaN values. Metrics may be incorrect.")

        return res


def _get_data_batch(
    input_batch: List[DataEntry],
    label_batch: List[DataEntry],
    forecast_batch: List[Forecast],
    seasonality: Optional[int] = None,
    mask_invalid_label: bool = True,
    allow_nan_forecast: bool = False,
) -> ChainMap:
    """
    Prepares a batch of data for evaluation by combining inputs, labels, and forecasts.
    
    This internal function processes batches of data to prepare them for evaluation metrics.
    It handles masking of invalid values and computes seasonal error for normalization.
    
    Args:
        input_batch: List of input data entries containing historical data
        label_batch: List of label data entries containing ground truth values
        forecast_batch: List of forecast objects containing model predictions
        seasonality: Optional seasonality period for seasonal error calculation
        mask_invalid_label: Whether to mask invalid values in labels
        allow_nan_forecast: Whether to allow NaN values in forecasts
    
    Returns:
        A ChainMap containing the processed data for evaluation metrics
    """
    # Stack label targets and mask invalid values if needed
    label_target = np.stack([label["target"] for label in label_batch], axis=0)
    if mask_invalid_label:
        label_target = np.ma.masked_invalid(label_target)

    other_data = {
        "label": label_target,
    }

    # Calculate seasonal error for each input
    seasonal_error_values = []
    for input_ in input_batch:
        seasonality_entry = seasonality
        if seasonality_entry is None:
            seasonality_entry = get_seasonality(input_["start"].freqstr)
        input_target = input_["target"]
        if mask_invalid_label:
            input_target = np.ma.masked_invalid(input_target)
        seasonal_error_values.append(
            seasonal_error(
                input_target,
                seasonality=seasonality_entry,
                time_axis=-1,
            )
        )
    other_data["seasonal_error"] = np.array(seasonal_error_values)

    # Combine data with batched forecasts
    return ChainMap(
        other_data, BatchForecast(forecast_batch, allow_nan=allow_nan_forecast)  # type: ignore
    )


def evaluate_forecasts_raw(
    forecasts: Iterable[Forecast],
    *,
    test_data: TestData,
    metrics,
    axis: Optional[Union[int, tuple]] = None,
    batch_size: int = 100,
    mask_invalid_label: bool = True,
    allow_nan_forecast: bool = False,
    seasonality: Optional[int] = None,
) -> dict:
    """
    Evaluate forecasts by comparing them with test data according to specified metrics.
    
    This function evaluates a set of forecasts against ground truth data using
    the provided metrics. It processes data in batches for efficiency and
    returns raw metric values.

    Args:
        forecasts: Iterable of Forecast objects to evaluate
        test_data: TestData object containing ground truth data
        metrics: List of metric definitions to compute
        axis: Controls aggregation of metrics:
            - None (default): aggregates across all dimensions
            - 0: aggregates across the dataset
            - 1: aggregates across time (univariate setting)
            - 2: aggregates across time (multivariate setting)
        batch_size: Number of items to process in each batch
        mask_invalid_label: Whether to mask invalid values in labels
        allow_nan_forecast: Whether to allow NaN values in forecasts
        seasonality: Optional seasonality period for seasonal error calculation
    
    Returns:
        Dictionary mapping metric names to their computed values
        
    Note:
        This feature is experimental and may be subject to changes.
    """
    # Determine dimensionality of the label data
    label_ndim = first(test_data.label)["target"].ndim
    assert label_ndim in [1, 2]

    # Process axis parameter
    if axis is None:
        axis = tuple(range(label_ndim + 1))
    if isinstance(axis, int):
        axis = (axis,)
    assert all(ax in range(3) for ax in axis)

    # Initialize evaluators for each metric
    evaluators = {}
    for metric in metrics:
        evaluator = metric(axis=axis)
        evaluators[evaluator.name] = evaluator

    index_data = []

    # Create batches for processing
    input_batches = batcher(test_data.input, batch_size=batch_size)
    label_batches = batcher(test_data.label, batch_size=batch_size)
    forecast_batches = batcher(forecasts, batch_size=batch_size)

    # Process each batch
    pbar = tqdm()
    for input_batch, label_batch, forecast_batch in zip(
        input_batches, label_batches, forecast_batches
    ):
        # Collect index data if needed
        if 0 not in axis:
            index_data.extend(
                [(forecast.item_id, forecast.start_date) for forecast in forecast_batch]
            )

        # Prepare data batch for evaluation
        data_batch = _get_data_batch(
            input_batch,
            label_batch,
            forecast_batch,
            seasonality=seasonality,
            mask_invalid_label=mask_invalid_label,
            allow_nan_forecast=allow_nan_forecast,
        )

        # Update each evaluator with the batch
        for evaluator in evaluators.values():
            evaluator.update(data_batch)

        pbar.update(len(forecast_batch))
    pbar.close()

    # Collect final metric values
    metrics_values = {
        metric_name: evaluator.get() for metric_name, evaluator in evaluators.items()
    }

    # Add index data if collected
    if index_data:
        metrics_values["__index_0"] = index_data

    return metrics_values


def evaluate_forecasts(
    forecasts: Iterable[Forecast],
    *,
    test_data: TestData,
    metrics,
    axis: Optional[Union[int, tuple]] = None,
    batch_size: int = 100,
    mask_invalid_label: bool = True,
    allow_nan_forecast: bool = False,
    seasonality: Optional[int] = None,
) -> pd.DataFrame:
    """
    Evaluate forecasts and return results as a Pandas DataFrame.
    
    This function is a wrapper around evaluate_forecasts_raw that formats
    the results as a Pandas DataFrame for easier analysis and visualization.
    
    Args:
        forecasts: Iterable of Forecast objects to evaluate
        test_data: TestData object containing ground truth data
        metrics: List of metric definitions to compute
        axis: Controls aggregation of metrics:
            - None (default): aggregates across all dimensions
            - 0: aggregates across the dataset
            - 1: aggregates across time (univariate setting)
            - 2: aggregates across time (multivariate setting)
        batch_size: Number of items to process in each batch
        mask_invalid_label: Whether to mask invalid values in labels
        allow_nan_forecast: Whether to allow NaN values in forecasts
        seasonality: Optional seasonality period for seasonal error calculation
    
    Returns:
        Pandas DataFrame containing the evaluation results, with metrics as columns
        
    Note:
        This feature is experimental and may be subject to changes.
    """
    # Get raw metric values
    metrics_values = evaluate_forecasts_raw(
        forecasts=forecasts,
        test_data=test_data,
        metrics=metrics,
        axis=axis,
        batch_size=batch_size,
        mask_invalid_label=mask_invalid_label,
        allow_nan_forecast=allow_nan_forecast,
        seasonality=seasonality,
    )
    # Extract index data if present
    index0 = metrics_values.pop("__index_0", None)

    # Create appropriate index for the DataFrame
    metric_shape = metrics_values[first(metrics_values)].shape
    if metric_shape == ():
        # Scalar metrics get a simple index
        index = [None]
    else:
        # Multi-dimensional metrics need a MultiIndex
        index_arrays = np.unravel_index(range(prod(metric_shape)), metric_shape)
        if index0 is not None:
            # Include item_id and start_date in the index if available
            index0_repeated = np.take(index0, indices=index_arrays[0], axis=0)
            index_arrays = (*zip(*index0_repeated), *index_arrays[1:])  # type: ignore
        index = pd.MultiIndex.from_arrays(index_arrays)

    # Flatten metric values for DataFrame format
    flattened_metrics = valmap(np.ravel, metrics_values)

    return pd.DataFrame(flattened_metrics, index=index)


def evaluate_model(
    model: Predictor,
    *,
    test_data: TestData,
    metrics,
    axis: Optional[Union[int, tuple]] = None,
    batch_size: int = 100,
    mask_invalid_label: bool = True,
    allow_nan_forecast: bool = False,
    seasonality: Optional[int] = None,
) -> pd.DataFrame:
    """
    Evaluate a forecasting model on test data.
    
    This function generates forecasts from the model using test data inputs,
    then evaluates those forecasts against the ground truth. It's a convenience
    wrapper that combines prediction and evaluation in one step.
    
    Args:
        model: Predictor object (forecasting model) to evaluate
        test_data: TestData object containing ground truth data
        metrics: List of metric definitions to compute
        axis: Controls aggregation of metrics:
            - None (default): aggregates across all dimensions
            - 0: aggregates across the dataset
            - 1: aggregates across time (univariate setting)
            - 2: aggregates across time (multivariate setting)
        batch_size: Number of items to process in each batch
        mask_invalid_label: Whether to mask invalid values in labels
        allow_nan_forecast: Whether to allow NaN values in forecasts
        seasonality: Optional seasonality period for seasonal error calculation
    
    Returns:
        Pandas DataFrame containing the evaluation results, with metrics as columns
        
    Note:
        This feature is experimental and may be subject to changes.
    """
    # Generate forecasts from the model
    forecasts = model.predict(test_data.input)

    # Evaluate the forecasts
    return evaluate_forecasts(
        forecasts=forecasts,
        test_data=test_data,
        metrics=metrics,
        axis=axis,
        batch_size=batch_size,
        mask_invalid_label=mask_invalid_label,
        allow_nan_forecast=allow_nan_forecast,
        seasonality=seasonality,
    )
