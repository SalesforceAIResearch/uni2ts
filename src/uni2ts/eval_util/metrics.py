"""
Metrics for evaluating time series forecasting models.

This module defines custom metrics for evaluating the performance of
forecasting models, building on the GluonTS evaluation framework.
"""

from dataclasses import dataclass
from functools import partial
from typing import Optional

from gluonts.ev.aggregations import Mean
from gluonts.ev.metrics import BaseMetricDefinition, DirectMetric
from gluonts.ev.stats import squared_error


@dataclass
class MedianMSE(BaseMetricDefinition):
    """
    Mean Squared Error metric for evaluating forecasts.
    
    This metric calculates the Mean Squared Error between the forecast
    and the actual values. By default, it uses the median (0.5 quantile)
    of the forecast distribution for comparison.
    
    Attributes:
        forecast_type: The forecast quantile to use for evaluation (default: "0.5")
                      This corresponds to the median of the forecast distribution.
    """

    forecast_type: str = "0.5"

    def __call__(self, axis: Optional[int] = None) -> DirectMetric:
        """
        Creates a DirectMetric for calculating Mean Squared Error.
        
        Args:
            axis: The axis along which to aggregate the metric.
                 None means aggregate over all dimensions.
        
        Returns:
            A DirectMetric object that computes MSE using the specified forecast type
            and aggregation axis.
        """
        return DirectMetric(
            name=f"MSE[{self.forecast_type}]",
            stat=partial(squared_error, forecast_type=self.forecast_type),
            aggregate=Mean(axis=axis),
        )
