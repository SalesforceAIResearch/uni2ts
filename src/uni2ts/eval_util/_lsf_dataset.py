"""
Long Sequence Forecasting (LSF) dataset adapter for time series evaluation.

This module provides a class for loading and processing various time series datasets
commonly used in long sequence forecasting research, making them compatible with
the evaluation utilities.
"""

import os

import numpy as np
import pandas as pd

from uni2ts.common.env import env


class LSFDataset:
    """
    Adapter for Long Sequence Forecasting datasets.
    
    This class loads various time series datasets commonly used in long sequence
    forecasting research and provides an iterator interface compatible with
    GluonTS evaluation utilities. It supports different modes for univariate
    and multivariate forecasting.
    
    Attributes:
        dataset_name: Name of the dataset
        mode: Mode for dataset loading ('S' for univariate, 'M' for multivariate,
              'MS' for multivariate with covariates)
        split: Dataset split ('train', 'val', or 'test')
        data: The loaded time series data
        length: Length of the dataset split
        start: Start timestamp of the time series
        freq: Time frequency of the dataset
        target_dim: Dimensionality of the target time series
        past_feat_dynamic_real_dim: Number of past dynamic real-valued features
    """
    
    def __init__(
        self,
        dataset_name: str,
        mode: str = "S",
        split: str = "test",
    ):
        """
        Initialize an LSFDataset adapter.
        
        Args:
            dataset_name: Name of the dataset to load (e.g., 'ETTh1', 'METR_LA')
            mode: Mode for dataset loading:
                  'S' - Univariate (single time series)
                  'M' - Multivariate (multiple time series as target)
                  'MS' - Multivariate with covariates (one target, others as features)
            split: Dataset split to use ('train', 'val', or 'test')
        
        Raises:
            ValueError: If dataset_name or mode is not recognized
        """
        self.dataset_name = dataset_name
        self.mode = mode
        self.split = split

        # Load the appropriate dataset based on name
        if dataset_name in ["ETTh1", "ETTh2"]:
            self._load_etth()
        elif dataset_name in ["ETTm1", "ETTm2"]:
            self._load_ettm()
        elif dataset_name == "METR_LA":
            self._load_metr_la()
        elif dataset_name == "solar":
            self._load_solar()
        elif dataset_name == "walmart":
            self._load_walmart()
        elif dataset_name == "electricity":
            self._load_custom("electricity/electricity.csv", "h")
        elif dataset_name == "weather":
            self._load_custom("weather/weather.csv", "10T")
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        # Set target dimensionality and feature dimensionality based on mode
        if mode == "S":
            # Univariate mode: single target, no features
            self.target_dim = 1
            self.past_feat_dynamic_real_dim = 0
        elif mode == "M":
            # Multivariate mode: all columns as targets
            self.target_dim = self.data.shape[-1]
            self.past_feat_dynamic_real_dim = 0
        elif mode == "MS":
            # Multivariate with covariates: one target, others as features
            self.target_dim = 1
            self.past_feat_dynamic_real_dim = self.data.shape[-1] - 1
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __iter__(self):
        """
        Iterate over samples in the dataset.
        
        The iteration behavior depends on the mode:
        - 'S' (Univariate): Yields each column as a separate univariate time series
        - 'M' (Multivariate): Yields a single multivariate time series with all columns
        - 'MS' (Multivariate with covariates): Yields each column as a target with
          all other columns as covariates
        
        Yields:
            Dictionary containing sample data with appropriate structure based on mode
        """
        if self.mode == "S":
            # Univariate mode: yield each column as a separate time series
            for i in range(self.data.shape[-1]):
                yield {
                    "target": self.data[:, i],
                    "start": self.start,
                }
        elif self.mode == "M":
            # Multivariate mode: yield all columns as a single multivariate time series
            yield {
                "target": self.data.transpose(1, 0),
                "start": self.start,
            }
        elif self.mode == "MS":
            # Multivariate with covariates: yield each column as target with others as features
            for i in range(self.data.shape[-1]):
                yield {
                    "target": self.data[:, i],
                    "past_feat_dynamic_real": np.concatenate(
                        [self.data[:, :i], self.data[:, i + 1 :]], axis=1
                    ).transpose(1, 0),
                    "start": self.start,
                }

    def scale(self, data, start, end):
        """
        Scale data using mean and standard deviation from a training segment.
        
        This method performs z-score normalization (standardization) using
        statistics computed from a specified segment of the data.
        
        Args:
            data: Data to scale
            start: Start index of the segment to compute statistics from
            end: End index of the segment to compute statistics from
            
        Returns:
            Scaled data with zero mean and unit variance (based on training segment)
        """
        train = data[start:end]
        mean = train.mean(axis=0)
        std = train.std(axis=0)
        return (data - mean) / std

    def _load_etth(self):
        """
        Load ETTh (Electricity Transformer Temperature - hourly) dataset.
        
        This method loads the ETTh1 or ETTh2 dataset, scales it, and
        prepares the appropriate split (train, val, or test).
        """
        # Load the CSV file
        df = pd.read_csv(
            os.path.join(env.LSF_PATH, f"ETT-small/{self.dataset_name}.csv")
        )

        # Define split lengths
        train_length = 8640
        val_length = 2880
        test_length = 2880
        
        # Scale the data using statistics from the training set
        data = self.scale(df[df.columns[1:]], 0, train_length).to_numpy()
        
        # Select the appropriate split
        if self.split == "train":
            self.data = data[:train_length]
            self.length = train_length
        elif self.split == "val":
            self.data = data[: train_length + val_length]
            self.length = val_length
        elif self.split == "test":
            self.data = data[: train_length + val_length + test_length]
            self.length = test_length
            
        # Set start time and frequency
        self.start = pd.to_datetime(df[["date"]].iloc[0].item())
        self.freq = "h"

    def _load_ettm(self):
        """
        Load ETTm (Electricity Transformer Temperature - 15min) dataset.
        
        This method loads the ETTm1 or ETTm2 dataset, scales it, and
        prepares the appropriate split (train, val, or test).
        """
        # Load the CSV file
        df = pd.read_csv(
            os.path.join(env.LSF_PATH, f"ETT-small/{self.dataset_name}.csv")
        )

        # Define split lengths
        train_length = 34560
        val_length = 11520
        test_length = 11520
        
        # Scale the data using statistics from the training set
        data = self.scale(df[df.columns[1:]], 0, train_length).to_numpy()
        
        # Select the appropriate split
        if self.split == "train":
            self.data = data[:train_length]
            self.length = train_length
        elif self.split == "val":
            self.data = data[: train_length + val_length]
            self.length = val_length
        elif self.split == "test":
            self.data = data[: train_length + val_length + test_length]
            self.length = test_length
            
        # Set start time and frequency
        self.start = pd.to_datetime(df[["date"]].iloc[0].item())
        self.freq = "15T"

    def _load_solar(self):
        """
        Load Solar dataset.
        
        This method loads the Solar dataset, reshapes and aggregates it,
        scales it, and prepares the appropriate split (train, val, or test).
        """
        # Load the text file
        df = pd.read_csv(os.path.join(env.LSF_PATH, "Solar/solar_AL.txt"), header=None)
        
        # Reshape and aggregate the data
        data = df.to_numpy().reshape(8760, 6, 137).sum(1)

        # Define split lengths (70% train, 10% val, 20% test)
        train_length = int(len(data) * 0.7)
        val_length = int(len(data) * 0.1)
        test_length = int(len(data) * 0.2)

        # Scale the data using statistics from the training set
        data = self.scale(data, 0, train_length).to_numpy()
        
        # Select the appropriate split
        if self.split == "train":
            self.data = data[:train_length]
            self.length = train_length
        elif self.split == "val":
            self.data = data[: train_length + val_length]
            self.length = val_length
        elif self.split == "test":
            self.data = data[: train_length + val_length + test_length]
            self.length = test_length
            
        # Set start time and frequency
        self.start = pd.to_datetime("2006-01-01")
        self.freq = "h"

    def _load_metr_la(self):
        """
        Load METR-LA (Los Angeles Metropolitan Traffic) dataset.
        
        This method loads the METR-LA traffic dataset, restructures it by entity,
        scales it, and prepares the appropriate split (train, val, or test).
        """
        # Load the dataset
        df = pd.read_csv(os.path.join(env.LSF_PATH, "METR_LA/METR_LA.dyna"))
        
        # Restructure data by entity ID
        data = []
        for id in df.entity_id.unique():
            id_df = df[df.entity_id == id]
            data.append(id_df.traffic_speed.to_numpy())
        data = np.stack(data, 1)

        # Define split lengths (70% train, 10% val, 20% test)
        train_length = int(len(data) * 0.7)
        val_length = int(len(data) * 0.1)
        test_length = int(len(data) * 0.2)

        # Scale the data using statistics from the training set
        data = self.scale(data, 0, train_length).to_numpy()
        
        # Select the appropriate split
        if self.split == "train":
            self.data = data[:train_length]
            self.length = train_length
        elif self.split == "val":
            self.data = data[: train_length + val_length]
            self.length = val_length
        elif self.split == "test":
            self.data = data[: train_length + val_length + test_length]
            self.length = test_length
            
        # Set start time and frequency
        self.start = pd.to_datetime("2012-03-01")
        self.freq = "5T"

    def _load_walmart(self):
        """
        Load Walmart store sales dataset.
        
        This method loads the Walmart sales dataset, restructures it by store and department,
        scales it, and prepares the appropriate split (train, val, or test).
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

        # Define split lengths
        train_length = 143 - 28 - 14  # Total - test - val
        val_length = 14
        test_length = 28
        
        # Scale the data using statistics from the training set
        data = self.scale(data, 0, train_length)
        
        # Select the appropriate split
        if self.split == "train":
            self.data = data[:train_length]
            self.length = train_length
        elif self.split == "val":
            self.data = data[: train_length + val_length]
            self.length = val_length
        elif self.split == "test":
            self.data = data[: train_length + val_length + test_length]
            self.length = test_length
            
        # Set start time and frequency
        self.start = pd.to_datetime("2010-02-05")
        self.freq = "W"

    def _load_custom(self, data_path: str, freq: str):
        """
        Load a custom dataset from a CSV file.
        
        This method loads a dataset from a specified CSV file, reorders columns,
        scales it, and prepares the appropriate split (train, val, or test).
        
        Args:
            data_path: Path to the CSV file relative to env.LSF_PATH
            freq: Time frequency of the dataset (e.g., 'h', '10T')
        """
        # Load the dataset
        df = pd.read_csv(os.path.join(env.LSF_PATH, data_path))
        
        # Reorder columns to put 'OT' at the end
        cols = list(df.columns)
        cols.remove("OT")
        cols.remove("date")
        df = df[["date"] + cols + ["OT"]]
        data = df[df.columns[1:]]

        # Define split lengths (70% train, 10% val, 20% test)
        train_length = int(len(data) * 0.7)
        val_length = int(len(data) * 0.1)
        test_length = int(len(data) * 0.2)
        
        # Scale the data using statistics from the training set
        data = self.scale(data, 0, train_length).to_numpy()
        
        # Select the appropriate split
        if self.split == "train":
            self.data = data[:train_length]
            self.length = train_length
        elif self.split == "val":
            self.data = data[: train_length + val_length]
            self.length = val_length
        elif self.split == "test":
            self.data = data[: train_length + val_length + test_length]
            self.length = test_length
            
        # Set start time and frequency
        self.start = pd.to_datetime(df[["date"]].iloc[0].item())
        self.freq = freq
