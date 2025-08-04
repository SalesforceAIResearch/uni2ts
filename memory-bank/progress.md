# Progress: Uni2TS Financial Model Fine-Tuning

## Current Status
We have developed a comprehensive strategy for training a Moirai model on multivariate OHLCV data. The strategy is documented in `docs/multivariate_ohlcv_strategy.md` and covers all aspects of the training pipeline, from data preparation to model evaluation.

## What Works
- The data loading and training pipeline is now understood.
- A pre-training job has been successfully run with an example dataset.
- The BTC dataset has been prepared and saved to disk.
- A detailed strategy document has been created for training models on multivariate OHLCV data.

## What's Left to Build
- **Data Preparation Pipeline**: Implement the data extraction, preprocessing, and feature engineering steps outlined in the strategy document.
- **Custom Dataset Builder**: Create a `FinancialDatasetBuilder` class that inherits from `datasets.builder.DatasetBuilder` to handle financial time series data.
- **Configuration Files**: Create the necessary configuration files for the data, model, and training process.
- **Training Execution**: Run the fine-tuning job using the CLI with the appropriate configuration.
- **Evaluation Scripts**: Develop scripts for evaluating the model's performance using the metrics defined in the strategy document.
- **Iterative Refinement**: Based on the evaluation results, refine the data preparation, model configuration, and training process.

## Known Issues
- None at this time.

## Evolution of Project Decisions
- **Initial Focus**: The project started with a general analysis of the repository and a plan for fine-tuning a model on a specific dataset (2015 BTC-USD 1-hour data).
- **Expanded Scope**: The scope has expanded to include a comprehensive strategy for training models on multivariate OHLCV data, with a focus on crypto assets initially.
- **Architectural Decisions**: The strategy document outlines key architectural decisions, such as the choice of patch size, context length, prediction length, and loss function, based on the Moirai model's capabilities and the characteristics of financial time series data.
- **Feature Engineering**: The strategy includes a detailed approach to feature engineering, including technical indicators, cross-asset features, and market regime indicators, to enhance the model's forecasting capabilities.
- **Evaluation Methodology**: The strategy defines a comprehensive evaluation methodology, with a focus on probabilistic forecasting metrics (CRPS, MSIS) and point forecast metrics (MSE, MAE), as well as practical metrics for trading applications.
