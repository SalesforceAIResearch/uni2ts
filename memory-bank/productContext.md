# Product Context: Uni2TS Financial Model Fine-Tuning

## Problem
Financial time series forecasting is a challenging task due to the complex, non-stationary, and often noisy nature of the data. Pre-trained time series models, like Moirai, offer a powerful foundation, but they need to be specialized to capture the specific patterns and dynamics of financial markets.

## Solution
By fine-tuning a pre-trained Moirai model on a large, high-quality dataset of financial pricing information, we aim to create a highly accurate multivariate forecasting model. This model will be tailored to the nuances of financial data, enabling more reliable predictions of future price movements.

## User Experience
The primary user of this project is a quantitative researcher or data scientist who needs to develop and evaluate time series forecasting models for financial applications. The desired experience is a streamlined and reproducible workflow that allows them to:

- Easily prepare custom financial datasets for use with the `uni2ts` library.
- Configure and launch fine-tuning experiments with minimal effort.
- Iterate on different model configurations and datasets to find the optimal solution.
- Have clear documentation of the process and project decisions.
