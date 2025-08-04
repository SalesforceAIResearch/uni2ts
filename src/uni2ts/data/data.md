# Data Module

The `data` module is responsible for loading, processing, and batching time series data. It includes classes for creating datasets, collating samples, and managing data loading pipelines.

## Files

### `dataset.py`

-   **`SampleTimeSeriesType`**: An enumeration that defines how to sample time series from a dataset (`NONE`, `UNIFORM`, `PROPORTIONAL`).
-   **`TimeSeriesDataset`**: A PyTorch `Dataset` for handling time series data. It wraps an `Indexer` and applies a transformation to the data.
-   **`MultiSampleTimeSeriesDataset`**: A `Dataset` that samples multiple time series and stacks them into a single sample.
-   **`EvalDataset`**: A `Dataset` class specifically for evaluation, which creates multiple evaluation windows for each time series.

### `loader.py`

-   **`Collate`**: An abstract base class for collate functions.
-   **`PadCollate`**: A collate function that pads uneven sequences to a `max_length`.
-   **`PackCollate`**: A collate function that packs uneven sequences using the first-fit-decreasing bin packing strategy.
-   **`SliceableBatchedSample`**: A wrapper around a `BatchedSample` that allows it to be sliced.
-   **`BatchedSampleQueue`**: A queue data structure for storing and managing batched samples.
-   **`_BatchedSampleIterator`**: An iterator that returns batched samples with a fixed batch size.
-   **`DataLoader`**: A wrapper around PyTorch's `DataLoader` that adds support for packing, cycling, and a fixed number of batches per epoch.

## Sub-modules

### `builder`

This sub-module contains builders for creating datasets from various sources.

### `indexer`

This sub-module provides classes for indexing and retrieving time series data.
