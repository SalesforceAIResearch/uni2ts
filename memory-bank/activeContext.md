# Active Context: Multivariate OHLCV Forecasting Strategy

## Current Focus
The current focus is on developing a comprehensive strategy for training a Moirai model on multivariate OHLCV (Open, High, Low, Close, Volume) data. We have created a detailed strategy document that outlines the data preparation pipeline, model configuration, training process, and evaluation methodology.

## Recent Changes
- Created a comprehensive strategy document (`docs/multivariate_ohlcv_strategy.md`) for training models on multivariate OHLCV data.
- Reviewed the codebase, focusing on the markdown files in all directories, especially those in `/home/dev/repos/uni2ts/src/uni2ts`.
- Gained a deep understanding of the Moirai model architecture and its capabilities for multivariate time series forecasting.
- Identified the key components of the uni2ts framework and how they can be leveraged for financial time series forecasting.

## Next Steps
1. **Implement the data preparation pipeline**: Create a script to extract OHLCV data from the Parquet data lake, preprocess it, and create a Hugging Face dataset.
2. **Create a custom `FinancialDatasetBuilder`**: Develop a specialized dataset builder for financial time series data.
3. **Configure the model for multivariate OHLCV data**: Set up the appropriate patch size, context length, prediction length, and other hyperparameters.
4. **Execute the training process**: Run the fine-tuning job using the CLI with the appropriate configuration.
5. **Evaluate the model**: Assess the model's performance using the evaluation utilities and metrics defined in the strategy document.
6. **Iterate and refine**: Based on the evaluation results, refine the data preparation, model configuration, and training process.

## Active Decisions
- **Data Selection**: We will focus on crypto assets (e.g., BTC, ETH) initially due to their 24/7 trading and cleaner data patterns.
- **Frequency**: We will use 1h data as a balance between signal granularity and sequence length.
- **Model Size**: We will start with the `moirai_1.1_R_base` model (91M parameters) for a balance of performance and computational requirements.
- **Patch Size**: For hourly data, we will use patch sizes of 32 or 64, as recommended in the MOIRAI documentation.
- **Context and Prediction Length**: For hourly data, we will use a context length of 168-336 hours (1-2 weeks) and a prediction length of 24-48 hours.
- **Loss Function**: We will use the negative log-likelihood (NLL) loss with a mixture distribution output for probabilistic forecasting.

## Important Patterns
- **Memory Bank**: All significant findings, decisions, and progress will be documented in the Memory Bank to ensure project continuity.
- **Iterative Approach**: We will start with a small, well-defined experiment and gradually increase complexity.
- **Data Preparation**: The strategy document provides a comprehensive pipeline for preparing OHLCV data, including handling missing values, outliers, and feature engineering.
- **Model Configuration**: The Moirai model's architecture is well-suited for multivariate time series forecasting, with its patchified masked encoder, any-variate attention, and mixture distribution output.
- **Training Process**: The uni2ts framework provides a streamlined process for training and fine-tuning models, with support for various optimizers, learning rate schedules, and loss functions.
- **Evaluation Methodology**: The evaluation utilities in the uni2ts framework allow for comprehensive assessment of model performance, with support for various metrics and visualization techniques.

## Learnings and Insights
- **Multivariate Handling**: The Moirai model's any-variate attention mechanism allows it to handle arbitrary numbers of variates with permutation-equivariance, making it well-suited for multivariate OHLCV data.
- **Patch Size Selection**: The choice of patch size is critical for performance, with larger patches reducing sequence length and computational cost, but potentially losing temporal detail.
- **Token Budget**: The token budget calculation (`tokens ≈ (#variates) × ⌈(context + horizon) / patch_size⌉`) is essential for ensuring that the model can handle the desired context and prediction lengths.
- **Feature Engineering**: Technical indicators, cross-asset features, and market regime indicators can significantly enhance the model's forecasting capabilities.
- **Probabilistic Forecasting**: The mixture distribution output allows for flexible probabilistic forecasting, capturing different aspects of the data distribution (heavy tails, skewness, etc.).
