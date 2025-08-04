# Active Context: Initial Setup and Planning

## Current Focus
The current focus is on configuring the BTC dataset for fine-tuning. We have successfully run a pre-training job with the `lib_city` dataset, which confirms our understanding of the data loading and training pipeline.

## Recent Changes
- Successfully ran a pre-training job using the `lib_city` dataset.
- Identified the need for a `.env` file and the `LOTSA_V1_PATH` environment variable.
- Created a data preparation script for the BTC dataset.
- Created the BTC dataset on disk.

## Next Steps
1.  **Explore the existing data preparation script**: Examine `example/prepare_financial_data.py` to understand how the BTC data is currently being processed.
2.  **Create a custom `DatasetBuilder`**: Create a new Python module, `src/uni2ts/data/builder/financial.py`, containing a `FinancialDatasetBuilder` class that inherits from `datasets.builder.DatasetBuilder`. This class will implement the `_info`, `_split_generators`, and `_generate_examples` methods to load the prepared BTC data from the parquet files.
3.  **Update `src/uni2ts/data/builder/__init__.py`**: Add `from .financial import FinancialDatasetBuilder` to make the new builder accessible.
4.  **Create a new data configuration file**: Create `cli/conf/finetune/data/financial_btc_2015.yaml` to define the configuration for the BTC dataset, pointing to the new `DatasetBuilder`.
5.  **Create a new fine-tuning run configuration file**: Create `cli/conf/finetune/finetune_btc.yaml`, modifying it to use the new data configuration (`financial_btc_2015`) and the `moirai_1.1_R_base` model.
6.  **Execute the fine-tuning job**: Run `python -m cli.train -cp conf/finetune run_name=btc_finetune model=moirai_1.1_R_base data=financial_btc_2015`.
7.  **Update the memory bank**: After the experiment, update `progress.md` and `activeContext.md` to document the results.

## Active Decisions
- **Initial Dataset**: We will start with 1-hour OHLCV data for BTC-USD from the year 2015. This provides a manageable subset for our first iteration.
- **Model**: We will use the `moirai_1.1_R_base` model as the starting point for fine-tuning.

## Important Patterns
- **Memory Bank**: All significant findings, decisions, and progress will be documented in the Memory Bank to ensure project continuity.
- **Iterative Approach**: We will start with a small, well-defined experiment and gradually increase complexity.
- **Data Preparation**: The `example/prepare_data.ipynb` notebook provides a clear template for creating Hugging Face datasets from custom data. The key is to create a generator function that yields dictionaries containing the time series data and metadata.
- **Forecasting Workflow**: The `example/moirai_forecast.ipynb` and `example/moirai_forecast_pandas.ipynb` notebooks demonstrate the end-to-end process of loading a pre-trained model, preparing data, and generating forecasts.
- **CLI for Training**: The `README.md` and the `cli` directory show that fine-tuning and evaluation are meant to be run from the command line using `python -m cli.train` and `python -m cli.eval` with Hydra for configuration.
