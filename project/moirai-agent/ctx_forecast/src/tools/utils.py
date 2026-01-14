import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
from pytz import timezone
from scipy.signal import find_peaks


def pad_to_left(
    series_list: list[list[int | float]], value=np.nan
) -> list[list[int | float]]:
    # assert series_list is a list of a list of float or int
    assert isinstance(series_list, list), "series_list must be a list"
    for d in series_list:
        assert isinstance(d, list), "d must be a list"
        for i in d:
            assert isinstance(i, (int, float)), "i must be a number"

    max_ctx_len = max(len(d) for d in series_list)
    output = [[value] * (max_ctx_len - len(d)) + d for d in series_list]
    return output


def plot_a_time_series(
    history_values: list[int | float],
    forecast_values: list[int | float] = None,
    future_values: list[int | float] = None,
    save_dir: str = "tmp",
    save_name: str = "plot",
) -> str:
    """
    Plot the given time-series sequences and return the path of the saved figure.
    Args:
        history_values: list, a list of numbers
        forecast_values: list, a list of numbers
        future_values: list, a list of numbers
        save_dir: str, the directory to save the figure
    Returns:
        save_path: str, the path of the saved figure
    """
    os.makedirs(save_dir, exist_ok=True)
    # check the input
    assert len(history_values) > 0, "history_values must be a non-empty list"
    if forecast_values is not None:
        assert len(forecast_values) > 0, "forecast_values must be a non-empty list"
    if future_values is not None:
        assert len(future_values) > 0, "future_values must be a non-empty list"
    if forecast_values is not None and future_values is not None:
        assert len(forecast_values) == len(
            future_values
        ), "forecast_values and future_values must have the same length"

    # plot the time series
    # plt.figure(figsize=(8, 4))

    history_indices = list(range(len(history_values)))
    plt.plot(
        history_indices, history_values, color="blue", linestyle="-", label="history"
    )
    # plt.plot(history_indices, history_values, color="blue", marker='o', linestyle=linestyle, label="history")

    if future_values is not None:
        future_start_idx = len(history_values)
        future_indices = list(
            range(future_start_idx, future_start_idx + len(future_values))
        )
        plt.plot(
            future_indices, future_values, color="red", linestyle="-", label="future"
        )
        # plt.plot(future_indices, future_values, color="red", marker='s', linestyle=linestyle, label="future")

    if forecast_values is not None:
        forecast_start_idx = len(history_values)
        forecast_indices = list(
            range(forecast_start_idx, forecast_start_idx + len(forecast_values))
        )
        plt.plot(
            forecast_indices,
            forecast_values,
            color="green",
            linestyle="-",
            label="forecast",
        )
        # plt.plot(forecast_indices, forecast_values, color="green", marker='^', linestyle=linestyle, label="forecast")

    # Add background color to the right side (future/forecast area)
    if future_values is not None or forecast_values is not None:
        separation_idx = (
            len(history_values) - 0.5
        )  # Position between last history and first future/forecast
        ax = plt.gca()
        ax.axvspan(separation_idx, ax.get_xlim()[1], alpha=0.3, color="gray", zorder=0)

    plt.gca().xaxis.set_major_locator(
        plt.MaxNLocator(integer=True)
    )  # Ensure x-axis shows integer indices
    plt.xlabel("relative index", fontweight="bold")
    plt.ylabel("value", fontweight="bold")
    plt.legend(fontsize=14)
    plt.grid()

    save_path = os.path.join(save_dir, save_name + ".jpg")
    # save_path = os.path.join(save_dir, save_name + "_" + datetime.now(timezone('Asia/Singapore')).strftime("%m%d-%H%M%S") + ".jpg")
    plt.savefig(save_path)
    plt.close()
    return save_path


def mean_absolute_error(
    ground_truth: list[int | float], prediction: list[int | float]
) -> float:
    """
    Given ground truth and prediction, calculate the mean absolute error.
    """
    # check the input
    assert isinstance(ground_truth, list), "ground_truth must be a list"
    assert isinstance(prediction, list), "prediction must be a list"
    assert len(ground_truth) > 0, "ground_truth must be a non-empty list"
    assert len(prediction) > 0, "prediction must be a non-empty list"
    assert len(ground_truth) == len(
        prediction
    ), "ground_truth and prediction must have the same length"

    # compute the mean absolute error
    ground_truth = np.array(ground_truth)
    prediction = np.array(prediction)
    error = np.mean(np.abs(ground_truth - prediction))
    return error


def time_series_pattern_analysis(time_series: list[int | float]) -> dict:
    """
    Given a time series, analyze the pattern of the time series and return the results.
    Detects common patterns including seasonality, trend, phase change, mean and std.

    Args:
        time_series: list of numbers representing the time series data

    Returns:
        dict: Analysis results containing detected patterns
    """
    # Check input
    assert isinstance(time_series, list), "time_series must be a list"
    assert len(time_series) > 0, "time_series must be a non-empty list"

    # Convert to numpy array for easier computation
    ts = np.array(time_series)
    n = len(ts)

    results = {}

    # Basic statistics
    results["mean"] = float(np.mean(ts))
    results["std"] = float(np.std(ts))
    results["min"] = float(np.min(ts))
    results["max"] = float(np.max(ts))
    results["range"] = results["max"] - results["min"]

    # Trend analysis using linear regression
    x = np.arange(n)
    slope, intercept = np.polyfit(x, ts, 1)
    results["trend"] = {
        "slope": float(slope),
        "direction": (
            "increasing"
            if slope > 0.01
            else "decreasing" if slope < -0.01 else "stable"
        ),
        "strength": (
            "strong"
            if abs(slope) > 0.1
            else "moderate" if abs(slope) > 0.01 else "weak"
        ),
    }

    # Seasonality detection using autocorrelation
    def autocorr(x, max_lag=None):
        if max_lag is None:
            max_lag = len(x) // 4
        autocorrs = []
        for lag in range(1, min(max_lag + 1, len(x))):
            if len(x) > lag:
                corr = np.corrcoef(x[:-lag], x[lag:])[0, 1]
                autocorrs.append(corr if not np.isnan(corr) else 0)
        return np.array(autocorrs)

    autocorrs = autocorr(ts)
    if len(autocorrs) > 0:
        # Find peaks in autocorrelation (potential seasonal periods)
        from scipy.signal import find_peaks

        peaks, _ = find_peaks(autocorrs, height=0.3, distance=2)

        if len(peaks) > 0:
            seasonal_periods = peaks + 1  # +1 because lag starts from 1
            results["seasonality"] = {
                "detected": True,
                "periods": seasonal_periods.tolist(),
                "max_autocorr": float(np.max(autocorrs[peaks])),
                "strength": (
                    "strong"
                    if np.max(autocorrs[peaks]) > 0.7
                    else "moderate" if np.max(autocorrs[peaks]) > 0.4 else "weak"
                ),
            }
        else:
            results["seasonality"] = {"detected": False}
    else:
        results["seasonality"] = {"detected": False}

    return results


if __name__ == "__main__":
    history_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10] * 10
    forecast_values = [11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
    future_values = [21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
    plot_a_time_series(history_values, forecast_values, future_values, save_dir="tmp")
    results = time_series_pattern_analysis(history_values)
    import ipdb

    ipdb.set_trace()
    pass
