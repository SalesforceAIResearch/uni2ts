"""
PF (Probabilistic Forecasting) dataset generation utilities.

This module provides functions for loading and generating datasets for various
time series forecasting benchmarks, making them compatible with the GluonTS
dataset format for evaluation.
"""

import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from gluonts.dataset import DatasetWriter
from gluonts.dataset.common import CategoricalFeatureInfo, MetaData, TrainDatasets

from uni2ts.common.env import env


def _load_etth(dataset_name: str, prediction_length: Optional[int] = None):
    """
    Load ETTh (Electricity Transformer Temperature - hourly) dataset for PF format.
    
    Args:
        dataset_name: Name of the ETTh dataset ('ETTh1' or 'ETTh2')
        prediction_length: Number of time steps to predict (default: 24 if None)
        
    Returns:
        Tuple containing:
        - data: The loaded time series data
        - start: Start timestamp of the time series
        - freq: Time frequency of the dataset ('h' for hourly)
        - prediction_length: Number of time steps to predict
        - rolling_evaluations: Number of rolling evaluation windows
    """
    # Load the CSV file
    df = pd.read_csv(os.path.join(env.LSF_PATH, f"ETT-small/{dataset_name}.csv"))
    data = df[df.columns[1:]].values

    # Extract metadata
    start = pd.to_datetime(df[["date"]].iloc[0].item())
    freq = "h"
    prediction_length = prediction_length or 24  # Default to 24 hours
    rolling_evaluations = 7  # One week of daily evaluations
    
    return data, start, freq, prediction_length, rolling_evaluations


def _load_ettm(dataset_name: str, prediction_length: Optional[int] = None):
    """
    Load ETTm (Electricity Transformer Temperature - 15min) dataset for PF format.
    
    Args:
        dataset_name: Name of the ETTm dataset ('ETTm1' or 'ETTm2')
        prediction_length: Number of time steps to predict (default: 96 if None)
        
    Returns:
        Tuple containing:
        - data: The loaded time series data
        - start: Start timestamp of the time series
        - freq: Time frequency of the dataset ('15T' for 15-minute)
        - prediction_length: Number of time steps to predict
        - rolling_evaluations: Number of rolling evaluation windows
    """
    # Load the CSV file
    df = pd.read_csv(os.path.join(env.LSF_PATH, f"ETT-small/{dataset_name}.csv"))
    data = df[df.columns[1:]].values

    # Extract metadata
    start = pd.to_datetime(df[["date"]].iloc[0].item())
    freq = "15T"
    prediction_length = prediction_length or 24 * 4  # Default to 24 hours (96 15-min intervals)
    rolling_evaluations = 7  # One week of daily evaluations
    
    return data, start, freq, prediction_length, rolling_evaluations


def _load_metr_la(dataset_name, prediction_length: Optional[int] = None):
    """
    Load METR-LA (Los Angeles Metropolitan Traffic) dataset for PF format.
    
    Args:
        dataset_name: Name of the dataset (unused, kept for API consistency)
        prediction_length: Number of time steps to predict (default: 288 if None)
        
    Returns:
        Tuple containing:
        - data: The loaded time series data
        - start: Start timestamp of the time series
        - freq: Time frequency of the dataset ('5T' for 5-minute)
        - prediction_length: Number of time steps to predict
        - rolling_evaluations: Number of rolling evaluation windows
    """
    # Load the dataset
    df = pd.read_csv(os.path.join(env.LSF_PATH, "METR_LA/METR_LA.dyna"))
    
    # Restructure data by entity ID
    data = []
    for id in df.entity_id.unique():
        id_df = df[df.entity_id == id]
        data.append(id_df.traffic_speed.to_numpy())
    data = np.stack(data, 1)

    # Extract metadata
    start = pd.to_datetime("2012-03-01")
    freq = "5T"
    prediction_length = prediction_length or 12 * 24  # Default to 24 hours (288 5-min intervals)
    rolling_evaluations = 7  # One week of daily evaluations
    
    return data, start, freq, prediction_length, rolling_evaluations


def _load_walmart(dataset_name: str, prediction_length: Optional[int] = None):
    """
    Load Walmart store sales dataset for PF format.
    
    Args:
        dataset_name: Name of the dataset (unused, kept for API consistency)
        prediction_length: Number of time steps to predict (default: 8 if None)
        
    Returns:
        Tuple containing:
        - data: The loaded time series data
        - start: Start timestamp of the time series
        - freq: Time frequency of the dataset ('W' for weekly)
        - prediction_length: Number of time steps to predict
        - rolling_evaluations: Number of rolling evaluation windows
    """
    # Load the dataset
    df = pd.read_csv(
        os.path.join(
            env.LSF_PATH, "walmart-recruiting-store-sales-forecasting/train.csv"
        )
    )

    # Restructure data by store and department
    data = []
    for id, row in df[["Store", "Dept"]].drop_duplicates().iterrows():
        row_df = df.query(f"Store == {row.Store} and Dept == {row.Dept}")
        # Only include complete time series
        if len(row_df) != 143:
            continue
        data.append(row_df.Weekly_Sales.to_numpy())
    data = np.stack(data, 1)

    # Extract metadata
    start = pd.to_datetime("2010-02-05")
    freq = "W"
    prediction_length = prediction_length or 8  # Default to 8 weeks
    rolling_evaluations = 4  # One month of weekly evaluations
    
    return data, start, freq, prediction_length, rolling_evaluations


def _load_jena_weather(dataset_name: str, prediction_length: Optional[int] = None):
    """
    Load Jena weather dataset for PF format.
    
    Args:
        dataset_name: Name of the dataset (unused, kept for API consistency)
        prediction_length: Number of time steps to predict (default: 144 if None)
        
    Returns:
        Tuple containing:
        - data: The loaded time series data
        - start: Start timestamp of the time series
        - freq: Time frequency of the dataset ('10T' for 10-minute)
        - prediction_length: Number of time steps to predict
        - rolling_evaluations: Number of rolling evaluation windows
    """
    # Load the dataset
    df = pd.read_csv(os.path.join(env.LSF_PATH, "weather/weather.csv"))
    
    # Reorder columns to put 'OT' at the end
    cols = list(df.columns)
    cols.remove("OT")
    cols.remove("date")
    df = df[["date"] + cols + ["OT"]]
    data = df[df.columns[1:]].to_numpy()

    # Extract metadata
    start = pd.to_datetime(df[["date"]].iloc[0].item())
    freq = "10T"
    prediction_length = prediction_length or 6 * 24  # Default to 24 hours (144 10-min intervals)
    rolling_evaluations = 7  # One week of daily evaluations
    
    return data, start, freq, prediction_length, rolling_evaluations


def _load_istanbul_traffic(dataset_name: str, prediction_length: Optional[int] = None):
    """
    Load Istanbul traffic dataset for PF format.
    
    Args:
        dataset_name: Name of the dataset (unused, kept for API consistency)
        prediction_length: Number of time steps to predict (default: 24 if None)
        
    Returns:
        Tuple containing:
        - data: The loaded time series data
        - start: Start timestamp of the time series
        - freq: Time frequency of the dataset ('h' for hourly)
        - prediction_length: Number of time steps to predict
        - rolling_evaluations: Number of rolling evaluation windows
    """
    # Load the dataset
    df = pd.read_csv(
        os.path.join(env.LSF_PATH, "istanbul-traffic-index/istanbul_traffic.csv")
    )
    
    # Convert to datetime and resample to hourly frequency
    df.datetime = pd.to_datetime(df.datetime)
    df = df.set_index("datetime")
    df = df.resample("h").mean()

    # Extract data and metadata
    data = df.values
    start = df.index[0]
    freq = "h"
    prediction_length = prediction_length or 24  # Default to 24 hours
    rolling_evaluations = 7  # One week of daily evaluations
    
    return data, start, freq, prediction_length, rolling_evaluations


def _load_turkey_power(dataset_name: str, prediction_length: Optional[int] = None):
    """
    Load Turkey power demand dataset for PF format.
    
    Args:
        dataset_name: Name of the dataset (unused, kept for API consistency)
        prediction_length: Number of time steps to predict (default: 24 if None)
        
    Returns:
        Tuple containing:
        - data: The loaded time series data
        - start: Start timestamp of the time series
        - freq: Time frequency of the dataset ('h' for hourly)
        - prediction_length: Number of time steps to predict
        - rolling_evaluations: Number of rolling evaluation windows
    """
    # Load the dataset
    df = pd.read_csv(
        os.path.join(
            env.LSF_PATH,
            "electrical-power-demand-in-turkey/power Generation and consumption.csv",
        )
    )
    
    # Convert to datetime and set as index
    df.Date_Time = pd.to_datetime(df.Date_Time, format="%d.%m.%Y %H:%M")
    df = df.set_index("Date_Time")

    # Extract data and metadata
    data = df.values
    start = df.index[0]
    freq = "h"
    prediction_length = prediction_length or 24  # Default to 24 hours
    rolling_evaluations = 7  # One week of daily evaluations
    
    return data, start, freq, prediction_length, rolling_evaluations


# Map of dataset names to their respective loading functions
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
    """
    Generate a GluonTS-compatible dataset for probabilistic forecasting.
    
    This function loads a dataset using the appropriate loading function,
    creates train and test time series with rolling evaluation windows,
    and saves the dataset in GluonTS format.
    
    Args:
        dataset_path: Path where the dataset will be saved
        dataset_name: Name of the dataset to load
        dataset_writer: GluonTS DatasetWriter to use for saving
        prediction_length: Number of time steps to predict (uses default if None)
    """
    # Load the dataset using the appropriate function
    load_func = pf_load_func_map[dataset_name]
    data, start, freq, prediction_length, rolling_evaluations = load_func(
        dataset_name, prediction_length
    )

    # Create training time series (one per category/variable)
    train_ts = []
    for cat in range(data.shape[-1]):
        # Exclude the last prediction_length * rolling_evaluations points for training
        sliced_ts = data[: -prediction_length * rolling_evaluations, cat]
        train_ts.append(
            {
                "target": sliced_ts,
                "start": start,
                "feat_static_cat": [cat],
                "item_id": cat,
            }
        )

    # Create test time series with rolling evaluation windows
    test_ts = []
    for window in range(rolling_evaluations - 1, -1, -1):
        for cat in range(data.shape[-1]):
            # Include data up to the current evaluation window
            sliced_ts = data[: len(data) - prediction_length * window, cat]
            test_ts.append(
                {
                    "target": sliced_ts,
                    "start": start,
                    "feat_static_cat": [cat],
                    "item_id": cat,
                }
            )

    # Create metadata for the dataset
    meta = MetaData(
        freq=freq,
        feat_static_cat=[
            CategoricalFeatureInfo(name="feat_static_cat_0", cardinality=data.shape[-1])
        ],
        prediction_length=prediction_length,
    )
    
    # Create and save the dataset
    dataset = TrainDatasets(metadata=meta, train=train_ts, test=test_ts)
    dataset.save(path_str=str(dataset_path), writer=dataset_writer, overwrite=True)
