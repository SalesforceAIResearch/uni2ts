# Tech Context: Uni2TS Fine-Tuning

## Technologies
- **Python**: The primary programming language.
- **PyTorch**: The deep learning framework used by `uni2ts`.
- **PyTorch Lightning**: For structuring the training code and simplifying the training loop.
- **Hydra**: For configuration management.
- **Polars**: For high-performance data manipulation of Parquet files.
- **Hugging Face `datasets`**: For creating and managing the time series datasets.
- **Parquet**: The columnar storage format for the financial data.
- **Git**: For version control.

## Development Setup
- The project is structured as a Python package with a `src` layout.
- A command-line interface is provided in the `cli` directory for training and evaluation.
- Example notebooks in the `example` directory demonstrate key functionalities.
- The development environment should have all the necessary Python packages installed, as specified in `pyproject.toml`.

## Dependencies
- `uni2ts`: The core time series modeling library.
- `torch`: The deep learning framework.
- `lightning`: The PyTorch training framework.
- `hydra-core`: The configuration management tool.
- `polars`: The data manipulation library.
- `datasets`: The dataset management library.
- `pyarrow`: For working with Parquet files.

## Tool Usage
- **Data Preparation**: Use `polars` to read and process Parquet files, and `datasets` to create the final dataset.
- **Training**: Use `python cli/train.py` with Hydra arguments to launch training jobs.
- **Configuration**: Create and modify YAML files in the `cli/conf` directory to define experiments.
