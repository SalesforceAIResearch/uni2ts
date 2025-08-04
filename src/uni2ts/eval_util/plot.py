"""
Plotting utilities for visualizing time series forecasts.

This module provides functions for plotting time series data along with
forecasts, making it easier to visually evaluate model performance.
"""

from typing import Iterator, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from gluonts import maybe
from gluonts.model import Forecast


def plot_single(
    inp: dict,
    label: dict,
    forecast: Forecast,
    context_length: int,
    intervals: tuple[float, ...] = (0.5, 0.9),
    ax: Optional[plt.axis] = None,
    dim: Optional[int] = None,
    name: Optional[str] = None,
    show_label: bool = False,
):
    """
    Plot a single time series with its forecast.
    
    This function visualizes a time series along with its forecast, showing both
    the historical context and the predicted future values. It can display prediction
    intervals to represent forecast uncertainty.
    
    Args:
        inp: Dictionary containing input data with historical time series
        label: Dictionary containing ground truth data for the forecast period
        forecast: Forecast object containing the model's predictions
        context_length: Number of historical time steps to show
        intervals: Tuple of quantile levels to plot as prediction intervals (default: (0.5, 0.9))
        ax: Matplotlib axis to plot on (creates one if None)
        dim: Dimension to plot for multivariate time series (None for univariate)
        name: Optional name for the forecast in the legend
        show_label: Whether to show labels in the legend for prediction intervals
    """
    # Get or create the matplotlib axis
    ax = maybe.unwrap_or_else(ax, plt.gca)

    # Concatenate historical and ground truth data
    target = np.concatenate([inp["target"], label["target"]], axis=-1)
    start = inp["start"]
    
    # Extract specific dimension for multivariate time series
    if dim is not None:
        target = target[dim]
        forecast = forecast.copy_dim(dim)

    # Create time index for the plot
    index = pd.period_range(start, periods=len(target), freq=start.freq)
    
    # Plot the target time series (historical + ground truth)
    ax.plot(
        index.to_timestamp()[-context_length - forecast.prediction_length :],
        target[-context_length - forecast.prediction_length :],
        label="target",
        color="black",
    )
    
    # Plot the forecast with prediction intervals
    forecast.plot(
        intervals=intervals,
        ax=ax,
        color="blue",
        name=name,
        show_label=show_label,
    )
    
    # Format the x-axis labels
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.legend(loc="lower left")


def plot_next_multi(
    axes: np.ndarray,
    input_it: Iterator[dict],
    label_it: Iterator[dict],
    forecast_it: Iterator[Forecast],
    context_length: int,
    intervals: tuple[float, ...] = (0.5, 0.9),
    dim: Optional[int] = None,
    name: Optional[str] = None,
    show_label: bool = False,
):
    """
    Plot multiple time series with their forecasts on a grid of axes.
    
    This function creates a multi-panel plot with each panel showing a different
    time series and its forecast. It's useful for comparing multiple forecasts
    or visualizing forecasts for different time series.
    
    Args:
        axes: Array of matplotlib axes to plot on
        input_it: Iterator of input dictionaries containing historical data
        label_it: Iterator of label dictionaries containing ground truth data
        forecast_it: Iterator of Forecast objects containing predictions
        context_length: Number of historical time steps to show
        intervals: Tuple of quantile levels to plot as prediction intervals (default: (0.5, 0.9))
        dim: Dimension to plot for multivariate time series (None for univariate)
        name: Optional name for the forecasts in the legends
        show_label: Whether to show labels in the legends for prediction intervals
    """
    # Ensure axes is a flat list
    axes = axes.flatten() if isinstance(axes, np.ndarray) else [axes]
    
    # Plot each time series and forecast on its corresponding axis
    for ax, inp, label, forecast in zip(axes, input_it, label_it, forecast_it):
        plot_single(
            inp,
            label,
            forecast,
            context_length,
            intervals=intervals,
            ax=ax,
            dim=dim,
            name=name,
            show_label=show_label,
        )
