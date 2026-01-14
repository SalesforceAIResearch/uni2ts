import base64
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# warnings.filterwarnings("ignore", message=".*set_ticklabels.*should only be used with a fixed number of ticks.*")


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def plot_ts_and_save(
    x,
    y,
    xlabel=None,
    xaxis_format=None,
    ylabel=None,
    title=None,
    figsize=None,
    grid=False,
    img_path="test.jpg",
    x_rotation=45,
):
    """
    Args:
        x: A sequence of x-values (e.g., timepoints).
        y: A sequence of y-values (e.g., signal values).
    """
    if not isinstance(x, (list, np.ndarray, pd.Series)) or not isinstance(
        y, (list, np.ndarray, pd.Series)
    ):
        return "Error: x and y must be list or numpy array"

    if len(x) != len(y):
        return f"Error: x and y must have the same length. Current length of x: {len(x)}, y: {len(y)}"

    try:
        if figsize is not None:
            fig = plt.figure(figsize=figsize)
        else:
            fig = plt.figure()  # Create a new figure

        ax = fig.add_subplot(1, 1, 1)
        ax.plot(x, y, linewidth=3.5, marker="o")

        if xaxis_format is None:
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d %H:%M:%S"))
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter(xaxis_format))

        if grid:
            ax.grid(True)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontsize=16)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontsize=16)
        if title is not None:
            ax.set_title(title)

        # ax.set_xticklabels(ax.get_xticklabels(), rotation=x_rotation)
        ax.tick_params(axis="x", labelsize=12, rotation=x_rotation)
        ax.tick_params(axis="y", labelsize=12)

        fig.savefig(img_path, bbox_inches="tight", pad_inches=0.1)
        plt.close(fig)
        return 1

    except Exception as e:
        raise e


def plot_historical_and_save(historical, img_path="test.jpg", y_pixels=448, dpi=100):
    # convert the historical to a pandas dataframe
    df = pd.DataFrame(historical)
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values(by="Date")

    x_values = df["Date"]
    y_values = df["Value"]

    x_pixels = y_pixels * min(3, max(1, len(x_values) // 10))
    figsize = (x_pixels / dpi, y_pixels / dpi)

    return plot_ts_and_save(
        x=x_values,
        y=y_values,
        xlabel=None,
        xaxis_format="%Y-%m-%d %H:%M",
        ylabel="Value",
        title=None,
        figsize=figsize,
        grid=True,
        img_path=img_path,
        x_rotation=45,
    )


if __name__ == "__main__":
    # generate a test case
    historical = {
        "Date": [
            "2024-01-01 00:00:00",
            "2024-01-01 01:00:00",
            "2024-01-01 02:00:00",
            "2024-01-01 03:00:00",
            "2024-01-01 04:00:00",
            "2024-01-01 05:00:00",
            "2024-01-01 06:00:00",
            "2024-01-01 07:00:00",
            "2024-01-01 08:00:00",
            "2024-01-01 09:00:00",
            "2024-01-01 10:00:00",
            "2024-01-01 11:00:00",
            "2024-01-01 12:00:00",
            "2024-01-01 13:00:00",
            "2024-01-01 14:00:00",
            "2024-01-01 15:00:00",
            "2024-01-01 16:00:00",
            "2024-01-01 17:00:00",
            "2024-01-01 18:00:00",
            "2024-01-01 19:00:00",
            "2024-01-01 20:00:00",
            "2024-01-01 21:00:00",
            "2024-01-01 22:00:00",
            "2024-01-01 23:00:00",
            "2024-01-02 00:00:00",
        ],
        "Value": [
            100,
            200,
            300,
            400,
            500,
            600,
            700,
            800,
            900,
            1000,
            1100,
            1200,
            1300,
            1400,
            1500,
            1600,
            1700,
            1800,
            1900,
            2000,
            2100,
            2200,
            2300,
            np.nan,
            2500,
        ],
    }
    img_path = "test.jpg"
    plot_historical_and_save(historical, img_path)
