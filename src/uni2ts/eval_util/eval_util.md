# Evaluation Utilities (eval_util)

## Overview

The `eval_util` module provides a comprehensive set of utilities for evaluating time series forecasting models. It handles dataset loading, model evaluation, metric calculation, and visualization of results. The module is designed to work with various dataset formats and supports different evaluation scenarios.

## Key Components

### Data Loading

The module provides functions to load datasets from different sources:

- **GluonTS datasets**: Standard time series datasets in GluonTS format
- **LSF datasets**: Long Sequence Forecasting datasets (ETT, METR-LA, etc.)
- **HF datasets**: Hugging Face datasets
- **PF datasets**: Probabilistic Forecasting datasets

These functions handle dataset preparation, splitting, and metadata creation to facilitate standardized model evaluation.

### Evaluation

The evaluation utilities allow for:

- Evaluating forecasts against ground truth data
- Computing various metrics with different aggregation options
- Evaluating models directly on test data
- Handling batch processing for efficient evaluation

### Metrics

The module defines custom metrics for evaluating forecasting performance:

- **MedianMSE**: Mean Squared Error using the median (0.5 quantile) of the forecast distribution

### Visualization

Plotting utilities are provided to visualize forecasts:

- Single time series with historical context and prediction intervals
- Multiple time series on a grid for comparison

## Files and Functions

### data.py

This file provides functions to load various types of datasets for evaluation.

Key functions:
- `get_gluonts_val_dataset`: Loads a validation dataset from GluonTS format
- `get_gluonts_test_dataset`: Loads a test dataset from GluonTS format
- `get_lsf_val_dataset`: Loads a validation dataset from LSF format
- `get_lsf_test_dataset`: Loads a test dataset from LSF format
- `get_custom_eval_dataset`: Creates a custom evaluation dataset with flexible parameters

Key classes:
- `MetaData`: Stores essential information about a dataset for evaluation

### evaluation.py

This file provides functions for evaluating forecasting models and their predictions.

Key functions:
- `evaluate_forecasts_raw`: Evaluates forecasts and returns raw metric values
- `evaluate_forecasts`: Evaluates forecasts and returns results as a DataFrame
- `evaluate_model`: Evaluates a model on test data

Key classes:
- `BatchForecast`: Wrapper around Forecast objects for batch processing

### metrics.py

This file defines custom metrics for evaluating forecasting performance.

Key classes:
- `MedianMSE`: Mean Squared Error metric using the median forecast

### plot.py

This file provides functions for visualizing time series forecasts.

Key functions:
- `plot_single`: Plots a single time series with its forecast
- `plot_next_multi`: Plots multiple time series with their forecasts on a grid

### _hf_dataset.py

This file provides a class for loading Hugging Face datasets.

Key classes:
- `HFDataset`: Adapter for Hugging Face datasets

### _lsf_dataset.py

This file provides a class for loading Long Sequence Forecasting datasets.

Key classes:
- `LSFDataset`: Adapter for LSF datasets with support for different modes (univariate, multivariate)

### _pf_dataset.py

This file provides functions for generating Probabilistic Forecasting datasets.

Key functions:
- `generate_pf_dataset`: Generates a GluonTS-compatible dataset for probabilistic forecasting
- Various dataset loading functions for different time series benchmarks

## Usage Examples

### Loading a Dataset

```python
from uni2ts.eval_util.data import get_lsf_test_dataset

# Load the ETTh1 dataset in univariate mode
test_data, metadata = get_lsf_test_dataset(
    dataset_name="ETTh1",
    prediction_length=24,
    mode="S"
)
```

### Evaluating a Model

```python
from uni2ts.eval_util.evaluation import evaluate_model
from gluonts.ev.metrics import MSE, CRPS

# Define metrics to evaluate
metrics = [MSE(), CRPS()]

# Evaluate the model
results = evaluate_model(
    model=my_forecasting_model,
    test_data=test_data,
    metrics=metrics
)

print(results)
```

### Visualizing Forecasts

```python
import matplotlib.pyplot as plt
from uni2ts.eval_util.plot import plot_single

# Create a figure
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# Plot a forecast
plot_single(
    inp=input_data,
    label=label_data,
    forecast=forecast,
    context_length=24,
    ax=ax
)

plt.show()
```

## Integration with Other Modules

The `eval_util` module is designed to work seamlessly with other components of the uni2ts framework:

- It uses dataset builders from the `data` module
- It leverages GluonTS evaluation utilities for metric calculation
- It supports models from the `model` module
- It can be used with the distribution types defined in the `distribution` module

This integration allows for a comprehensive evaluation pipeline for time series forecasting models.
