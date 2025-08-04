"""
Data loading utilities for evaluation of time series forecasting models.

This module provides functions to load various types of datasets (GluonTS, LSF, custom)
for evaluation purposes. It handles dataset preparation, splitting, and metadata creation
to facilitate standardized model evaluation across different dataset types.
"""

from functools import partial
from typing import NamedTuple

import gluonts
from gluonts.dataset.common import _FileDataset
from gluonts.dataset.split import TestData, split

from uni2ts.data.builder.lotsa_v1.gluonts import get_dataset

from ._hf_dataset import HFDataset
from ._lsf_dataset import LSFDataset
from ._pf_dataset import generate_pf_dataset, pf_load_func_map

# Register PF datasets with GluonTS repository
# This allows PF datasets to be accessed through the GluonTS dataset interface
gluonts.dataset.repository.dataset_recipes |= {
    k: partial(generate_pf_dataset, dataset_name=k) for k in pf_load_func_map.keys()
}


class MetaData(NamedTuple):
    """
    Metadata for time series datasets used in evaluation.
    
    This class stores essential information about a dataset that is needed
    for proper evaluation of forecasting models.
    
    Attributes:
        freq: The time frequency of the dataset (e.g., 'h' for hourly, 'D' for daily)
        target_dim: Dimensionality of the target time series (1 for univariate)
        prediction_length: Number of time steps to predict
        feat_dynamic_real_dim: Number of dynamic real-valued features
        past_feat_dynamic_real_dim: Number of past dynamic real-valued features
        split: Dataset split identifier ('train', 'val', or 'test')
    """
    freq: str
    target_dim: int
    prediction_length: int
    feat_dynamic_real_dim: int = 0
    past_feat_dynamic_real_dim: int = 0
    split: str = "test"


def get_gluonts_val_dataset(
    dataset_name: str,
    prediction_length: int = None,
    mode: str = None,
    regenerate: bool = False,
) -> tuple[TestData, MetaData]:
    """
    Retrieves a validation dataset from GluonTS format for evaluation.
    
    This function loads a dataset from GluonTS, splits it to create a validation set,
    and returns both the dataset and its metadata for model evaluation.
    
    Args:
        dataset_name: Name of the dataset to load
        prediction_length: Number of time steps to predict (uses default if None)
        mode: Mode for dataset loading (unused but kept for API consistency)
        regenerate: Whether to regenerate the dataset instead of using cached version
    
    Returns:
        A tuple containing:
        - test_data: The validation dataset in GluonTS TestData format
        - metadata: Metadata about the dataset (frequency, dimensions, etc.)
    """
    # Default prediction lengths for known datasets
    default_prediction_lengths = {
        "australian_electricity_demand": 336,
        "pedestrian_counts": 24,
    }
    if prediction_length is None and dataset_name in default_prediction_lengths:
        prediction_length = default_prediction_lengths[dataset_name]

    # Load the dataset
    dataset = get_dataset(
        dataset_name, prediction_length=prediction_length, regenerate=regenerate
    )

    # Use dataset's prediction length if not specified
    prediction_length = prediction_length or dataset.metadata.prediction_length
    
    # Split the dataset and generate validation instances
    _, test_template = split(dataset.train, offset=-prediction_length)
    test_data = test_template.generate_instances(prediction_length)
    
    # Create metadata for the validation dataset
    metadata = MetaData(
        freq=dataset.metadata.freq,
        target_dim=1,
        prediction_length=prediction_length,
        split="val",
    )
    return test_data, metadata


def get_gluonts_test_dataset(
    dataset_name: str,
    prediction_length: int = None,
    mode: str = None,
    regenerate: bool = False,
) -> tuple[TestData, MetaData]:
    """
    Retrieves a test dataset from GluonTS format for evaluation.
    
    Similar to get_gluonts_val_dataset but creates a test set instead of validation set.
    This function loads a dataset from GluonTS, splits it to create a test set,
    and returns both the dataset and its metadata for model evaluation.
    
    Args:
        dataset_name: Name of the dataset to load
        prediction_length: Number of time steps to predict (uses default if None)
        mode: Mode for dataset loading (unused but kept for API consistency)
        regenerate: Whether to regenerate the dataset instead of using cached version
    
    Returns:
        A tuple containing:
        - test_data: The test dataset in GluonTS TestData format
        - metadata: Metadata about the dataset (frequency, dimensions, etc.)
    """
    # Default prediction lengths for known datasets
    default_prediction_lengths = {
        "australian_electricity_demand": 336,
        "pedestrian_counts": 24,
    }
    if prediction_length is None and dataset_name in default_prediction_lengths:
        prediction_length = default_prediction_lengths[dataset_name]

    # Load the dataset
    dataset = get_dataset(
        dataset_name, prediction_length=prediction_length, regenerate=regenerate
    )

    # Use dataset's prediction length if not specified
    prediction_length = prediction_length or dataset.metadata.prediction_length
    
    # Split the dataset and generate test instances
    _, test_template = split(dataset.test, offset=-prediction_length)
    test_data = test_template.generate_instances(prediction_length)
    
    # Create metadata for the test dataset
    metadata = MetaData(
        freq=dataset.metadata.freq,
        target_dim=1,
        prediction_length=prediction_length,
        split="test",
    )
    return test_data, metadata


def get_lsf_val_dataset(
    dataset_name: str,
    prediction_length: int = 96,
    mode: str = "S",
) -> tuple[TestData, MetaData]:
    """
    Retrieves a validation dataset from LSF (Long Sequence Forecasting) format.
    
    This function loads an LSF dataset, converts it to GluonTS format,
    splits it to create a validation set, and returns both the dataset
    and its metadata for model evaluation.
    
    Args:
        dataset_name: Name of the LSF dataset to load (e.g., 'ETTh1', 'METR_LA')
        prediction_length: Number of time steps to predict (default: 96)
        mode: Dataset mode - 'S' for univariate, 'M' for multivariate,
              'MS' for multivariate with covariates
    
    Returns:
        A tuple containing:
        - test_data: The validation dataset in GluonTS TestData format
        - metadata: Metadata about the dataset (frequency, dimensions, etc.)
    """
    # Load the LSF dataset in validation split
    lsf_dataset = LSFDataset(dataset_name, mode=mode, split="val")
    
    # Convert to GluonTS FileDataset format
    dataset = _FileDataset(
        lsf_dataset, freq=lsf_dataset.freq, one_dim_target=lsf_dataset.target_dim == 1
    )
    
    # Split the dataset and generate validation instances
    _, test_template = split(dataset, offset=-lsf_dataset.length)
    test_data = test_template.generate_instances(
        prediction_length,
        windows=lsf_dataset.length - prediction_length + 1,
        distance=1,
    )
    
    # Create metadata for the validation dataset
    metadata = MetaData(
        freq=lsf_dataset.freq,
        target_dim=lsf_dataset.target_dim,
        prediction_length=prediction_length,
        past_feat_dynamic_real_dim=lsf_dataset.past_feat_dynamic_real_dim,
        split="val",
    )
    return test_data, metadata


def get_lsf_test_dataset(
    dataset_name: str,
    prediction_length: int = 96,
    mode: str = "S",
) -> tuple[TestData, MetaData]:
    """
    Retrieves a test dataset from LSF (Long Sequence Forecasting) format.
    
    Similar to get_lsf_val_dataset but creates a test set instead of validation set.
    This function loads an LSF dataset, converts it to GluonTS format,
    splits it to create a test set, and returns both the dataset
    and its metadata for model evaluation.
    
    Args:
        dataset_name: Name of the LSF dataset to load (e.g., 'ETTh1', 'METR_LA')
        prediction_length: Number of time steps to predict (default: 96)
        mode: Dataset mode - 'S' for univariate, 'M' for multivariate,
              'MS' for multivariate with covariates
    
    Returns:
        A tuple containing:
        - test_data: The test dataset in GluonTS TestData format
        - metadata: Metadata about the dataset (frequency, dimensions, etc.)
    """
    # Load the LSF dataset in test split
    lsf_dataset = LSFDataset(dataset_name, mode=mode, split="test")
    
    # Convert to GluonTS FileDataset format
    dataset = _FileDataset(
        lsf_dataset, freq=lsf_dataset.freq, one_dim_target=lsf_dataset.target_dim == 1
    )
    
    # Split the dataset and generate test instances
    _, test_template = split(dataset, offset=-lsf_dataset.length)
    test_data = test_template.generate_instances(
        prediction_length,
        windows=lsf_dataset.length - prediction_length + 1,
        distance=1,
    )
    
    # Create metadata for the test dataset
    metadata = MetaData(
        freq=lsf_dataset.freq,
        target_dim=lsf_dataset.target_dim,
        prediction_length=prediction_length,
        past_feat_dynamic_real_dim=lsf_dataset.past_feat_dynamic_real_dim,
        split="test",
    )
    return test_data, metadata


def get_custom_eval_dataset(
    dataset_name: str,
    offset: int,
    windows: int,
    distance: int,
    prediction_length: int,
    mode: None = None,
) -> tuple[TestData, MetaData]:
    """
    Creates a custom evaluation dataset from a Hugging Face dataset.
    
    This function provides more flexibility in dataset creation by allowing
    custom parameters for offset, windows, and distance between evaluation instances.
    
    Args:
        dataset_name: Name of the Hugging Face dataset to load
        offset: Number of time steps to offset from the end of the dataset
        windows: Number of evaluation windows to generate
        distance: Distance between consecutive evaluation windows
        prediction_length: Number of time steps to predict
        mode: Mode for dataset loading (unused but kept for API consistency)
    
    Returns:
        A tuple containing:
        - test_data: The custom dataset in GluonTS TestData format
        - metadata: Metadata about the dataset (frequency, dimensions, etc.)
    """
    # Load the Hugging Face dataset
    hf_dataset = HFDataset(dataset_name)
    
    # Convert to GluonTS FileDataset format
    dataset = _FileDataset(
        hf_dataset, freq=hf_dataset.freq, one_dim_target=hf_dataset.target_dim == 1
    )
    
    # Split the dataset with custom offset and generate test instances
    _, test_template = split(dataset, offset=offset)
    test_data = test_template.generate_instances(
        prediction_length,
        windows=windows,
        distance=distance,
    )
    
    # Create metadata for the custom dataset
    metadata = MetaData(
        freq=hf_dataset.freq,
        target_dim=hf_dataset.target_dim,
        prediction_length=prediction_length,
        split="test",
    )
    return test_data, metadata
