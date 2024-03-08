import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from gluonts.dataset import DatasetWriter
from gluonts.dataset.common import CategoricalFeatureInfo, MetaData, TrainDatasets

from uni2ts.common.env import env


def _load_etth(dataset_name: str, prediction_length: Optional[int] = None):
    df = pd.read_csv(os.path.join(env.LSF_PATH, f"ETT-small/{dataset_name}.csv"))
    data = df[df.columns[1:]].values

    start = pd.to_datetime(df[["date"]].iloc[0].item())
    freq = "h"
    prediction_length = prediction_length or 24
    rolling_evaluations = 7
    return data, start, freq, prediction_length, rolling_evaluations


def _load_ettm(dataset_name: str, prediction_length: Optional[int] = None):
    df = pd.read_csv(os.path.join(env.LSF_PATH, f"ETT-small/{dataset_name}.csv"))
    data = df[df.columns[1:]].values

    start = pd.to_datetime(df[["date"]].iloc[0].item())
    freq = "15T"
    prediction_length = prediction_length or 24 * 4
    rolling_evaluations = 7
    return data, start, freq, prediction_length, rolling_evaluations


def _load_metr_la(dataset_name, prediction_length: Optional[int] = None):
    df = pd.read_csv(os.path.join(env.LSF_PATH, "METR_LA/METR_LA.dyna"))
    data = []
    for id in df.entity_id.unique():
        id_df = df[df.entity_id == id]
        data.append(id_df.traffic_speed.to_numpy())
    data = np.stack(data, 1)

    start = pd.to_datetime("2012-03-01")
    freq = "5T"
    prediction_length = prediction_length or 12 * 24
    rolling_evaluations = 7
    return data, start, freq, prediction_length, rolling_evaluations


def _load_walmart(dataset_name: str, prediction_length: Optional[int] = None):
    df = pd.read_csv(
        os.path.join(
            env.LSF_PATH, "walmart-recruiting-store-sales-forecasting/train.csv"
        )
    )

    data = []
    for id, row in df[["Store", "Dept"]].drop_duplicates().iterrows():
        row_df = df.query(f"Store == {row.Store} and Dept == {row.Dept}")
        if len(row_df) != 143:
            continue
        data.append(row_df.Weekly_Sales.to_numpy())
    data = np.stack(data, 1)

    start = pd.to_datetime("2010-02-05")
    freq = "W"
    prediction_length = prediction_length or 8
    rolling_evaluations = 4
    return data, start, freq, prediction_length, rolling_evaluations


def _load_jena_weather(dataset_name: str, prediction_length: Optional[int] = None):
    df = pd.read_csv(os.path.join(env.LSF_PATH, "weather/weather.csv"))
    cols = list(df.columns)
    cols.remove("OT")
    cols.remove("date")
    df = df[["date"] + cols + ["OT"]]
    data = df[df.columns[1:]].to_numpy()

    start = pd.to_datetime(df[["date"]].iloc[0].item())
    freq = "10T"
    prediction_length = prediction_length or 6 * 24
    rolling_evaluations = 7
    return data, start, freq, prediction_length, rolling_evaluations


def _load_istanbul_traffic(dataset_name: str, prediction_length: Optional[int] = None):
    df = pd.read_csv(
        os.path.join(env.LSF_PATH, "istanbul-traffic-index/istanbul_traffic.csv")
    )
    df.datetime = pd.to_datetime(df.datetime)
    df = df.set_index("datetime")
    df = df.resample("h").mean()

    data = df.values
    start = df.index[0]
    freq = "h"
    prediction_length = prediction_length or 24
    rolling_evaluations = 7
    return data, start, freq, prediction_length, rolling_evaluations


def _load_turkey_power(dataset_name: str, prediction_length: Optional[int] = None):
    df = pd.read_csv(
        os.path.join(
            env.LSF_PATH,
            "electrical-power-demand-in-turkey/power Generation and consumption.csv",
        )
    )
    df.Date_Time = pd.to_datetime(df.Date_Time, format="%d.%m.%Y %H:%M")
    df = df.set_index("Date_Time")

    data = df.values
    start = df.index[0]
    freq = "h"
    prediction_length = prediction_length or 24
    rolling_evaluations = 7
    return data, start, freq, prediction_length, rolling_evaluations


pf_load_func_map = {
    "ETTh1": _load_etth,
    "ETTh2": _load_etth,
    "ETTm1": _load_ettm,
    "ETTm2": _load_ettm,
    "METR_LA": _load_metr_la,
    "walmart": _load_walmart,
    "jena_weather": _load_jena_weather,
    "istanbul_traffic": _load_istanbul_traffic,
    "turkey_power": _load_turkey_power,
}


def generate_pf_dataset(
    dataset_path: Path,
    dataset_name: str,
    dataset_writer: DatasetWriter,
    prediction_length: Optional[int] = None,
):
    load_func = pf_load_func_map[dataset_name]
    data, start, freq, prediction_length, rolling_evaluations = load_func(
        dataset_name, prediction_length
    )

    train_ts = []
    for cat in range(data.shape[-1]):
        sliced_ts = data[: -prediction_length * rolling_evaluations, cat]
        train_ts.append(
            {
                "target": sliced_ts,
                "start": start,
                "feat_static_cat": [cat],
                "item_id": cat,
            }
        )

    test_ts = []
    for window in range(rolling_evaluations - 1, -1, -1):
        for cat in range(data.shape[-1]):
            sliced_ts = data[: len(data) - prediction_length * window, cat]
            test_ts.append(
                {
                    "target": sliced_ts,
                    "start": start,
                    "feat_static_cat": [cat],
                    "item_id": cat,
                }
            )

    meta = MetaData(
        freq=freq,
        feat_static_cat=[
            CategoricalFeatureInfo(name="feat_static_cat_0", cardinality=data.shape[-1])
        ],
        prediction_length=prediction_length,
    )
    dataset = TrainDatasets(metadata=meta, train=train_ts, test=test_ts)
    dataset.save(path_str=str(dataset_path), writer=dataset_writer, overwrite=True)
