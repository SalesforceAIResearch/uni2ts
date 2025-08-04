# Indexer Module

The `indexer` module provides classes for indexing and retrieving time series data from various sources.

## Files

### `_base.py`

-   **`Indexer`**: An abstract base class for all indexers. It defines a common interface for accessing data and providing a sequence-like interface to it.

### `hf_dataset_indexer.py`

-   **`HuggingFaceDatasetIndexer`**: An indexer for Hugging Face datasets. It provides an interface for accessing data from a `datasets.Dataset` object.
