# Builder Module

The `builder` module provides classes for building and loading datasets from various sources.

## Files

### `_base.py`

-   **`DatasetBuilder`**: An abstract base class for dataset builders.
-   **`ConcatDatasetBuilder`**: A dataset builder that concatenates multiple `DatasetBuilder` instances.

### `financial.py`

-   **`FinancialDatasetBuilder`**: A dataset builder for financial time series data.

### `simple.py`

-   **`_from_long_dataframe`**: Converts a long format DataFrame to a generator function and HuggingFace Features.
-   **`_from_wide_dataframe`**: Converts a wide format DataFrame to a generator function and HuggingFace Features.
-   **`_from_wide_dataframe_multivariate`**: Converts a wide format multivariate DataFrame to a generator function and HuggingFace Features.
-   **`SimpleDatasetBuilder`**: A simple dataset builder that can build datasets from CSV files.
-   **`SimpleEvalDatasetBuilder`**: A simple evaluation dataset builder that can build datasets from CSV files.
-   **`generate_eval_builders`**: Generates a list of `SimpleEvalDatasetBuilder` instances for a given set of evaluation parameters.

## Sub-modules

### `lotsa_v1`

This sub-module contains builders for the LOTSA-v1 dataset.
