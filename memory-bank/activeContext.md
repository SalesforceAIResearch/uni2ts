# Active Context: Multivariate OHLCV Forecasting Strategy and Pre-training

## Current Focus
The current focus is on developing a comprehensive strategy for training a Moirai model on multivariate OHLCV (Open, High, Low, Close, Volume) data. We have created a detailed strategy document that outlines the data preparation pipeline, model configuration, training process, and evaluation methodology. Additionally, we have implemented a solution for pre-training the MOIRAI model on large financial datasets stored in parquet format.

## Recent Changes
- Created a comprehensive strategy document (`docs/multivariate_ohlcv_strategy.md`) for training models on multivariate OHLCV data
- Reviewed the codebase, focusing on the markdown files in all directories, especially those in `/home/dev/repos/uni2ts/src/uni2ts`
- Gained a deep understanding of the Moirai model architecture and its capabilities for multivariate time series forecasting
- Identified the key components of the uni2ts framework and how they can be leveraged for financial time series forecasting
- Implemented a custom `IterativeFinancialDatasetBuilder` for processing large financial datasets iteratively
- Created configuration files for pre-training the MOIRAI model on financial data
- Developed test scripts and training scripts for the pre-training process
- Created documentation for the pre-training implementation
- **NEW**: Successfully created financial dataset from parquet files (BTC 2015-2020)
- **NEW**: Verified dataset loading and structure compatibility with MOIRAI
- **NEW**: Successfully tested single epoch fine-tuning without validation
- **NEW**: Confirmed that validation is properly disabled when no val_data is specified

## Next Steps
1. **Test the pre-training implementation**: Run the test script to verify that the dataset builder works correctly
2. **Execute the pre-training process**: Run the pre-training script to train the MOIRAI model on financial data
3. **Monitor the pre-training process**: Use TensorBoard to monitor the training metrics and adjust hyperparameters as needed
4. **Evaluate the pre-trained model**: Assess the model's performance on financial forecasting tasks
5. **Fine-tune the pre-trained model**: Use the pre-trained model as a starting point for fine-tuning on specific forecasting tasks
6. **Iterate and refine**: Based on the evaluation results, refine the data preparation, model configuration, and training process
7. **Expand to other assets**: After successful BTC training, expand to other crypto currencies, forex, equities, etc.

## Active Decisions
- **Data Selection**: We will focus on crypto assets (e.g., BTC, ETH) initially due to their 24/7 trading and cleaner data patterns
- **Frequency**: We will use 1h data as a balance between signal granularity and sequence length
- **Model Size**: We will start with the `moirai_1.1_R_base` model (91M parameters) for a balance of performance and computational requirements
- **Patch Size**: For hourly data, we will use patch sizes of [16, 32, 64] for pre-training, as recommended in the MOIRAI documentation
- **Context and Prediction Length**: For hourly data, we will use a context length of 168-336 hours (1-2 weeks) and a prediction length of 24-48 hours
- **Loss Function**: We will use the negative log-likelihood (NLL) loss with a mixture distribution output for probabilistic forecasting
- **Pre-training Approach**: We will use an iterative approach to process large financial datasets, avoiding loading the entire dataset into memory at once
- **Batch Processing**: We will process assets in batches, with a default batch size of 10, to manage memory usage and enable processing of very large datasets
- **Validation Strategy**: **NEW**: Validation will be disabled during training by not specifying val_data parameter, allowing for training similar to pre-training approach
- **Dataset Creation**: **NEW**: Successfully implemented proper dataset creation from parquet files using custom builder

## Important Patterns
- **Memory Bank**: All significant findings, decisions, and progress will be documented in the Memory Bank to ensure project continuity
- **Iterative Approach**: We will start with a small, well-defined experiment and gradually increase complexity
- **Data Preparation**: The strategy document provides a comprehensive pipeline for preparing OHLCV data, including handling missing values, outliers, and feature engineering
- **Model Configuration**: The Moirai model's architecture is well-suited for multivariate time series forecasting, with its patchified masked encoder, any-variate attention, and mixture distribution output
- **Training Process**: The uni2ts framework provides a streamlined process for training and fine-tuning models, with support for various optimizers, learning rate schedules, and loss functions
- **Evaluation Methodology**: The evaluation utilities in the uni2ts framework allow for comprehensive assessment of model performance, with support for various metrics and visualization techniques
- **Iterative Data Processing**: Our custom dataset builder processes data iteratively, one asset at a time, to avoid loading the entire dataset into memory at once
- **Temporary Storage Management**: We use temporary storage for processed data, cleaning up after each batch to manage disk space
- **Validation Disable**: **NEW**: Successfully implemented approach to disable validation by not specifying val_data parameter

## Learnings and Insights
- **Multivariate Handling**: The Moirai model's any-variate attention mechanism allows it to handle arbitrary numbers of variates with permutation-equivariance, making it well-suited for multivariate OHLCV data
- **Patch Size Selection**: The choice of patch size is critical for performance, with larger patches reducing sequence length and computational cost, but potentially losing temporal detail
- **Token Budget**: The token budget calculation (`tokens ≈ (#variates) × ⌈(context + horizon) / patch_size⌉`) is essential for ensuring that the model can handle the desired context and prediction lengths
- **Feature Engineering**: Technical indicators, cross-asset features, and market regime indicators can significantly enhance the model's forecasting capabilities
- **Probabilistic Forecasting**: The mixture distribution output allows for flexible probabilistic forecasting, capturing different aspects of the data distribution (heavy tails, skewness, etc.)
- **Pre-training vs. Fine-tuning**: Pre-training on a large, diverse dataset can significantly improve the model's performance on downstream tasks, even with limited task-specific data
- **Memory Management**: Processing large datasets requires careful memory management, especially when working with time series data that can have complex structures and dependencies
- **Iterative Processing**: Breaking down large datasets into manageable chunks and processing them iteratively can significantly reduce memory usage and enable processing of very large datasets
- **Validation Disable**: **NEW**: Learned that validation can be completely disabled by not specifying val_data parameter in configuration
- **Dataset Compatibility**: **NEW**: Confirmed that OHLCV data from parquet files can be properly formatted for MOIRAI training
