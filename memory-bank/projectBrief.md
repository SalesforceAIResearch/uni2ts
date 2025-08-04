# Project Brief: Uni2TS Financial Model Fine-Tuning for Multivariate OHLCV Data

## Project Name
Uni2TS Financial Model Fine-Tuning for Multivariate OHLCV Data

## Objective
Fine-tune a pre-trained Moirai model from the `uni2ts` library on a large dataset of multivariate OHLCV (Open, High, Low, Close, Volume) data to create a specialized forecasting model for financial time series.

## Scope
- Analyze the `uni2ts` repository to understand its mechanics for training and fine-tuning.
- Develop a comprehensive data preparation pipeline for multivariate OHLCV data stored in a Parquet data lake.
- Implement feature engineering techniques to enhance the model's forecasting capabilities.
- Configure and run a fine-tuning experiment on a subset of the financial data.
- Evaluate the model's performance using both probabilistic and point forecast metrics.
- Establish a repeatable workflow for fine-tuning with different datasets and model configurations.
- Document the entire process in a detailed strategy document.

## Key Deliverables
- A comprehensive strategy document for training models on multivariate OHLCV data.
- A data preparation script for extracting, preprocessing, and feature engineering of OHLCV data.
- A custom `FinancialDatasetBuilder` class for creating Hugging Face datasets from OHLCV data.
- Hydra configuration files for the fine-tuning experiment.
- Evaluation scripts for assessing model performance.
- A well-documented process for fine-tuning models within this repository.
- A set of memory bank files to ensure project continuity.

## Success Criteria
- The model achieves competitive performance on financial time series forecasting tasks.
- The data preparation pipeline successfully handles the complexities of OHLCV data.
- The fine-tuning process is well-documented and reproducible.
- The evaluation framework provides a comprehensive assessment of the model's performance.
- The memory bank files provide a clear record of the project's progress and decisions.

## Timeline
1. **Week 1**: Analyze the repository, understand the data structure, and create the strategy document.
2. **Week 2**: Implement the data preparation pipeline and create the custom dataset builder.
3. **Week 3**: Configure and run the fine-tuning experiment.
4. **Week 4**: Evaluate the model's performance and refine the approach.
5. **Week 5**: Document the process and finalize the memory bank files.

## Resources
- The `uni2ts` repository and its documentation.
- Pre-trained Moirai models from Salesforce AI Research.
- A Parquet data lake containing OHLCV data for various assets.
- Computational resources for training and fine-tuning models.
- Python libraries for data manipulation, feature engineering, and evaluation.
