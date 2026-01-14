import base64
import random
import re
import warnings

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

DEFAULT_SPECIAL_TAGS = {
    "start_history_values": "<history>",
    "end_history_values": "</history>",
    "start_future_values": "\\boxed{",
    "end_future_values": "}",
    "entry_sep": ",",
    "missing_value": "N/A",
    "start_context": "<context>",
    "end_context": "</context>",
}


def set_random_seeds(seed: int = 42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def parse_values_from_string(
    content: str, start_tag: str = "\\boxed{", end_tag: str = "}", entry_sep: str = ","
) -> list[float]:
    start_tag = start_tag.replace(
        "\\", "\\\\"
    )  # escape the special characters in the start_tag and end_tag
    pattern = re.compile(r"{}(.*?){}".format(start_tag, end_tag), re.DOTALL)
    results = re.findall(pattern, content)
    if not results:
        return []
    else:
        results = results[-1]  # using the last one
        entries = (
            results.split(start_tag)[-1].split(end_tag)[0].strip(" \n").split(entry_sep)
        )
        values = []
        for entry in entries:
            try:
                if entry.startswith("[") and entry.endswith("]"):
                    entry = entry[1:-1].split(",")[-1].strip()
                v = float(entry)
                values.append(v)
            except:
                values.append(np.nan)
        return values


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


def plot_predictions_and_save(
    df_history,
    df_preds,
    df_crossval_preds=None,
    img_path="test.jpg",
    grid=True,
    x_rotation=45,
):

    model_names = list(df_preds.keys())
    x_all = pd.concat([df_history["Date"]] + [df_preds[model_names[0]]["Date"]])
    x_min, x_max, len_total = x_all.min(), x_all.max(), len(x_all)
    # y_pixels, dpi = 448, 100
    y_pixels, dpi = 896, 200
    x_pixels = y_pixels * min(3, max(1, len_total // 10))

    all_pred_lines = [
        {
            "legend": "history",
            "x": df_history["Date"],
            "y": df_history["Value"],
            "marker": "o",
        }
    ]
    for model_name in model_names:
        all_pred_lines.append(
            {
                "legend": model_name,
                "x": df_preds[model_name]["Date"],
                "y": df_preds[model_name]["Value"],
                "marker": "o",
            }
        )

    if df_crossval_preds is not None:
        all_crsval_lines = [
            {
                "legend": "history",
                "x": df_history["Date"],
                "y": df_history["Value"],
                "marker": "o",
            }
        ]
        for model_name in model_names:
            all_crsval_lines.append(
                {
                    "legend": model_name,
                    "x": df_crossval_preds[model_name]["Date"],
                    "y": df_crossval_preds[model_name]["Value"],
                    "marker": "^",
                }
            )

    if df_crossval_preds is None:
        fig = plt.figure(figsize=(x_pixels / dpi, y_pixels / dpi))
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("Predictions", fontsize=16)
        for i, line in enumerate(all_pred_lines):
            # ax.plot(line['x'], line['y'], marker=line['marker'], label=line['legend'], color=f'C{i}')
            ax.plot(
                line["x"], line["y"], label=line["legend"], color=f"C{i}", linewidth=2.0
            )
    else:
        fig = plt.figure(figsize=(x_pixels / dpi, y_pixels / dpi * 2))

        ax = fig.add_subplot(2, 1, 1)
        ax.set_title("Predictions", fontsize=16)
        for i, line in enumerate(all_pred_lines):
            # ax.plot(line['x'], line['y'], marker=line['marker'], label=line['legend'], color=f'C{i}')
            ax.plot(
                line["x"], line["y"], label=line["legend"], color=f"C{i}", linewidth=2.0
            )

        plt.subplots_adjust(hspace=0.5)

        ax = fig.add_subplot(2, 1, 2)
        ax.set_title("Cross-validation", fontsize=16)
        for i, line in enumerate(all_crsval_lines):
            # ax.plot(line['x'], line['y'], marker=line['marker'], label=line['legend'], color=f'C{i}')
            ax.plot(
                line["x"], line["y"], label=line["legend"], color=f"C{i}", linewidth=2.0
            )

    # if xaxis_format is None:
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M:%S'))
    # else:
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter(xaxis_format))

    axs = fig.get_axes()
    for ax in axs:
        ax.set_xlim(x_min, x_max)
        ax.grid(grid)
        ax.tick_params(axis="x", labelsize=12, rotation=x_rotation)
        ax.tick_params(axis="y", labelsize=12)
        ax.legend(fontsize=14)

    fig.savefig(img_path, bbox_inches="tight", pad_inches=0.1, dpi=dpi)
    plt.close(fig)


if __name__ == "__main__":
    # generate a test case
    historical = {
        "Date": [
            "2023-12-31 12:00:00",
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
    df_history = pd.DataFrame(historical)
    df_history["Date"] = pd.to_datetime(df_history["Date"])
    df_history = df_history.sort_values(by="Date")
    predictions = {
        "model1": {
            "Date": [
                "2024-01-02 01:00:00",
                "2024-01-02 02:00:00",
                "2024-01-02 03:00:00",
                "2024-01-02 04:00:00",
                "2024-01-02 05:00:00",
                "2024-01-02 06:00:00",
                "2024-01-02 07:00:00",
                "2024-01-02 08:00:00",
                "2024-01-02 09:00:00",
                "2024-01-02 10:00:00",
                "2024-01-02 11:00:00",
                "2024-01-02 12:00:00",
                "2024-01-02 13:00:00",
                "2024-01-02 14:00:00",
                "2024-01-02 15:00:00",
                "2024-01-02 16:00:00",
                "2024-01-02 17:00:00",
                "2024-01-02 18:00:00",
                "2024-01-02 19:00:00",
                "2024-01-02 20:00:00",
                "2024-01-02 21:00:00",
                "2024-01-02 22:00:00",
                "2024-01-02 23:00:00",
                "2024-01-03 00:00:00",
            ],
            "Value": [
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
        },
        "model2": {
            "Date": [
                "2024-01-02 01:00:00",
                "2024-01-02 02:00:00",
                "2024-01-02 03:00:00",
                "2024-01-02 04:00:00",
                "2024-01-02 05:00:00",
                "2024-01-02 06:00:00",
                "2024-01-02 07:00:00",
                "2024-01-02 08:00:00",
                "2024-01-02 09:00:00",
                "2024-01-02 10:00:00",
                "2024-01-02 11:00:00",
                "2024-01-02 12:00:00",
                "2024-01-02 13:00:00",
                "2024-01-02 14:00:00",
                "2024-01-02 15:00:00",
                "2024-01-02 16:00:00",
                "2024-01-02 17:00:00",
                "2024-01-02 18:00:00",
                "2024-01-02 19:00:00",
                "2024-01-02 20:00:00",
                "2024-01-02 21:00:00",
                "2024-01-02 22:00:00",
                "2024-01-02 23:00:00",
                "2024-01-03 00:00:00",
            ],
            "Value": [
                250,
                350,
                450,
                550,
                650,
                750,
                850,
                950,
                1050,
                1150,
                1250,
                1350,
                1450,
                1550,
                1650,
                1750,
                1850,
                1950,
                2050,
                2150,
                2250,
                2350,
                np.nan,
                2500,
            ],
        },
    }
    df_preds = {}
    for model_name in predictions:
        df_preds[model_name] = pd.DataFrame(predictions[model_name])
        df_preds[model_name]["Date"] = pd.to_datetime(df_preds[model_name]["Date"])
        df_preds[model_name] = df_preds[model_name].sort_values(by="Date")

    crossval_preds = {
        "model1": {
            "Date": [
                "2024-01-01 19:00:00",
                "2024-01-01 20:00:00",
                "2024-01-01 21:00:00",
                "2024-01-01 22:00:00",
                "2024-01-01 23:00:00",
                "2024-01-02 00:00:00",
            ],
            "Value": [200, 300, 400, 500, 600, 700],
        },
        "model2": {
            "Date": [
                "2024-01-01 19:00:00",
                "2024-01-01 20:00:00",
                "2024-01-01 21:00:00",
                "2024-01-01 22:00:00",
                "2024-01-01 23:00:00",
                "2024-01-02 00:00:00",
            ],
            "Value": [250, 350, 450, 550, 650, 750],
        },
    }
    df_crossval_preds = {}
    for model_name in crossval_preds:
        df_crossval_preds[model_name] = pd.DataFrame(crossval_preds[model_name])
        df_crossval_preds[model_name]["Date"] = pd.to_datetime(
            df_crossval_preds[model_name]["Date"]
        )
        df_crossval_preds[model_name] = df_crossval_preds[model_name].sort_values(
            by="Date"
        )

    img_path = "tmp1.png"
    plot_predictions_and_save(df_history, df_preds, df_crossval_preds, img_path)
