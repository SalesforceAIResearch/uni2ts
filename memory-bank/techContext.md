# Tech Context: Uni2TS Fine-Tuning for Multivariate OHLCV Data

## Technologies
- **Python**: The primary programming language.
- **PyTorch**: The deep learning framework used by `uni2ts`.
- **PyTorch Lightning**: For structuring the training code and simplifying the training loop.
- **Hydra**: For configuration management.
- **Polars**: For high-performance data manipulation of Parquet files.
- **Pandas**: For data manipulation and technical indicator calculation.
- **Hugging Face `datasets`**: For creating and managing the time series datasets.
- **Parquet**: The columnar storage format for the financial data.
- **Git**: For version control.
- **Matplotlib**: For visualization of forecasts and evaluation metrics.
- **scikit-learn**: For feature selection and normalization.

## Development Setup
- The project is structured as a Python package with a `src` layout.
- A command-line interface is provided in the `cli` directory for training and evaluation.
- Example notebooks in the `example` directory demonstrate key functionalities.
- The development environment should have all the necessary Python packages installed, as specified in `pyproject.toml`.
- The `docs` directory contains strategy documents and other documentation.

## Dependencies
- `uni2ts`: The core time series modeling library.
- `torch`: The deep learning framework.
- `lightning`: The PyTorch training framework.
- `hydra-core`: The configuration management tool.
- `polars`: The data manipulation library.
- `pandas`: The data analysis library.
- `datasets`: The dataset management library.
- `pyarrow`: For working with Parquet files.
- `matplotlib`: For visualization.
- `scikit-learn`: For machine learning utilities.
- `ta-lib` or `pandas-ta`: For technical indicator calculation.

## Tool Usage
- **Data Preparation**: 
  - Use `polars` to read and process Parquet files.
  - Use `pandas` for technical indicator calculation and feature engineering.
  - Use `datasets` to create the final dataset with the appropriate schema for uni2ts.

- **Training**: 
  - Use `python cli/train.py` with Hydra arguments to launch training jobs.
  - Configure the optimizer, learning rate schedule, and loss function.
  - Use PyTorch Lightning callbacks for checkpointing and early stopping.

- **Configuration**: 
  - Create and modify YAML files in the `cli/conf` directory to define experiments.
  - Use Hydra's composition capabilities to combine different configurations.
  - Override specific parameters via command-line arguments.

- **Evaluation**: 
  - Use the evaluation utilities in `uni2ts.eval_util` to assess model performance.
  - Calculate probabilistic and point forecast metrics.
  - Create visualizations of forecasts with confidence intervals.

## Multivariate OHLCV Data Handling
- **Data Structure**: 
  - OHLCV data is stored in a Parquet data lake with a hive-style partitioning scheme.
  - The partitioning follows the pattern: `asset_class=X/freq=Y/symbol=Z/year=YYYY/month=MM/part.parquet`.
  - Each Parquet file contains the columns: `ts`, `open`, `high`, `low`, `close`, `volume`.

- **Data Processing**: 
  - Use `polars` for efficient reading and processing of Parquet files.
  - Handle missing values, outliers, and apply normalization.
  - Add technical indicators, cross-asset features, and market regime indicators.

- **Dataset Creation**: 
  - Create a Hugging Face dataset with the appropriate schema for uni2ts.
  - Structure the data as a multivariate time series with shape (variates, time).
  - Include metadata such as frequency, start timestamp, and item identifier.

- **Feature Engineering**: 
  - Calculate technical indicators using `pandas-ta` or a similar library.
  - Add cross-asset features to capture relationships between different assets.
  - Create market regime indicators to identify different market conditions.

## Model Configuration
- **Patch Size**: 
  - For hourly data, use patch sizes of 32 or 64.
  - The choice of patch size balances temporal detail and computational efficiency.

- **Context and Prediction Length**: 
  - For hourly data, use a context length of 168-336 hours (1-2 weeks) and a prediction length of 24-48 hours.
  - Ensure that the context length is at least 5x the prediction length for stable forecasting.

- **Token Budget**: 
  - Calculate the token budget using the formula: `tokens ≈ (#variates) × ⌈(context + horizon) / patch_size⌉`.
  - Ensure that the total token count stays within the model's limits (typically 512 tokens in pre-training).

- **Distribution Type**: 
  - Use a mixture distribution output for flexible probabilistic forecasting.
  - The mixture can include Student-t, Negative Binomial, Log-Normal, and Normal distributions.
