"""
Hugging Face dataset adapter for time series evaluation.

This module provides a class for loading and iterating over time series datasets
stored in Hugging Face's datasets format, making them compatible with the
evaluation utilities.
"""

from pathlib import Path

import datasets

from uni2ts.common.env import env


class HFDataset:
    """
    Adapter for Hugging Face datasets to be used with time series evaluation utilities.
    
    This class loads a dataset from disk in Hugging Face's datasets format and
    provides an iterator interface compatible with GluonTS evaluation utilities.
    It extracts metadata such as frequency and target dimensionality from the dataset.
    
    Attributes:
        hf_dataset: The loaded Hugging Face dataset
        freq: The time frequency of the dataset
        target_dim: Dimensionality of the target time series (1 for univariate)
    """
    
    def __init__(self, dataset_name: str, storage_path: Path = env.CUSTOM_DATA_PATH):
        """
        Initialize a HFDataset adapter.
        
        Args:
            dataset_name: Name of the dataset to load
            storage_path: Path to the directory containing the dataset (default: env.CUSTOM_DATA_PATH)
        """
        # Load the dataset from disk and convert to numpy format
        self.hf_dataset = datasets.load_from_disk(
            str(storage_path / dataset_name)
        ).with_format("numpy")
        
        # Extract frequency from the first sample
        self.freq = self.hf_dataset[0]["freq"]
        
        # Determine target dimensionality (1 for univariate, >1 for multivariate)
        self.target_dim = (
            target.shape[-1]
            if len((target := self.hf_dataset[0]["target"]).shape) > 1
            else 1
        )

    def __iter__(self):
        """
        Iterate over samples in the dataset.
        
        Yields:
            Dictionary containing sample data with start time converted to a Python object
        """
        for sample in self.hf_dataset:
            # Convert start time to a Python object (needed for GluonTS compatibility)
            sample["start"] = sample["start"].item()
            yield sample
