# Progress: Uni2TS Financial Model Pre-training and Fine-Tuning

## Current Status
We have successfully developed and implemented a comprehensive solution for training a Moirai model on multivariate OHLCV data. The implementation allows for both pre-training and fine-tuning approaches on large financial datasets stored in parquet format.

## What Works
- ✅ The data loading and training pipeline is now understood and working
- ✅ A pre-training job has been successfully run with an example dataset
- ✅ The BTC dataset has been prepared and saved to disk
- ✅ A detailed strategy document has been created for training models on multivariate OHLCV data
- ✅ A custom `IterativeFinancialDatasetBuilder` has been implemented for processing large financial datasets iteratively
- ✅ Configuration files for pre-training the MOIRAI model on financial data have been created
- ✅ Test scripts and training scripts for the pre-training process have been developed
- ✅ Documentation for the pre-training implementation has been created
- ✅ **NEW**: Successfully created financial dataset from parquet files (BTC 2015-2020)
- ✅ **NEW**: Verified dataset loading and structure compatibility with MOIRAI
- ✅ **NEW**: Successfully tested single epoch fine-tuning without validation
- ✅ **NEW**: Confirmed that validation is properly disabled when no val_data is specified

## What's Left to Build
- **Test Pre-training Implementation**: Run the test script to verify that the dataset builder works correctly
- **Execute Pre-training**: Run the pre-training script to train the MOIRAI model on financial data
- **Monitor Pre-training**: Use TensorBoard to monitor the training metrics and adjust hyperparameters as needed
- **Evaluate Pre-trained Model**: Assess the model's performance on financial forecasting tasks
- **Fine-tune Pre-trained Model**: Use the pre-trained model as a starting point for fine-tuning on specific forecasting tasks
- **Iterative Refinement**: Based on the evaluation results, refine the data preparation, model configuration, and training process

## Known Issues
- The pre-training implementation has not been tested yet, so there may be issues with the dataset builder or configuration files
- The pre-training process may require significant computational resources, especially for large datasets

## Evolution of Project Decisions
- **Initial Focus**: The project started with a general analysis of the repository and a plan for fine-tuning a model on a specific dataset (2015 BTC-USD 1-hour data)
- **Expanded Scope**: The scope has expanded to include a comprehensive strategy for training models on multivariate OHLCV data, with a focus on crypto assets initially
- **Architectural Decisions**: The strategy document outlines key architectural decisions, such as the choice of patch size, context length, prediction length, and loss function, based on the Moirai model's capabilities and the characteristics of financial time series data
- **Feature Engineering**: The strategy includes a detailed approach to feature engineering, including technical indicators, cross-asset features, and market regime indicators, to enhance the model's forecasting capabilities
- **Evaluation Methodology**: The strategy defines a comprehensive evaluation methodology, with a focus on probabilistic forecasting metrics (CRPS, MSIS) and point forecast metrics (MSE, MAE), as well as practical metrics for trading applications
- **Pre-training Approach**: We have decided to implement a custom dataset builder that processes financial data iteratively, one asset at a time, to avoid loading the entire dataset into memory at once. This approach allows us to pre-train the MOIRAI model on large financial datasets without requiring excessive memory or storage
- **Batch Processing**: We have decided to process assets in batches, with a default batch size of 10, to manage memory usage and enable processing of very large datasets
- **Validation Strategy**: **NEW**: Successfully implemented approach to disable validation during training by not specifying val_data parameter
- **Dataset Creation**: **NEW**: Successfully created proper dataset format compatible with MOIRAI training pipeline
