# Loss Module

The `loss` module provides a comprehensive collection of loss functions for training and evaluating time series forecasting models in the uni2ts framework. These loss functions are specifically designed to handle the challenges of time series data, including variable-length sequences, missing values, and probabilistic forecasts.

## Core Components

### Packed Loss Architecture

The module is built around a "packed" architecture, which efficiently handles variable-length time series batched together. This architecture has several key components:

- **PackedLoss**: Abstract base class for all packed loss functions, handling masking, sample identification, and proper loss reduction.
- **PackedPointLoss**: Base class for losses that evaluate point forecasts (deterministic predictions).
- **PackedDistributionLoss**: Base class for losses that evaluate probabilistic forecasts (distribution predictions).

### Loss Function Categories

The loss functions are organized into several categories:

#### Point Losses

These losses evaluate deterministic forecasts by comparing predicted values to actual values:

- **PackedMAELoss**: Mean Absolute Error, which measures the average absolute difference between predicted and actual values.
- **PackedMSELoss**: Mean Squared Error, which measures the average squared difference between predicted and actual values.
- **PackedRMSELoss**: Root Mean Squared Error, which is the square root of MSE and provides a measure in the same units as the target variable.

#### Distribution Losses

These losses evaluate probabilistic forecasts by assessing the likelihood of the actual values under the predicted distribution:

- **PackedNLLLoss**: Negative Log-Likelihood Loss, which computes the negative log-likelihood of the target values under the predicted probability distribution.

#### Normalized Losses

These losses incorporate normalization to make them more comparable across different scales:

- **PackedNMAELoss**: Normalized Mean Absolute Error, which normalizes MAE by a reference value.
- **PackedNMSELoss**: Normalized Mean Squared Error, which normalizes MSE by a reference value.
- **PackedNRMSELoss**: Normalized Root Mean Squared Error, which normalizes RMSE by a reference value.
- **PackedNMLSELoss**: Normalized Mean Log Squared Error, which applies a log transformation before computing MSE.

#### Percentage Error Losses

These losses express errors as percentages of the actual values:

- **PackedMAPELoss**: Mean Absolute Percentage Error, which measures the average absolute percentage difference between predicted and actual values.
- **PackedSMAPELoss**: Symmetric Mean Absolute Percentage Error, which is a symmetric version of MAPE that treats over-predictions and under-predictions equally.

## Key Features

### Masking Support

All loss functions support masking for handling:
- Missing values in the target data (observed_mask)
- Prediction windows vs. context windows (prediction_mask)
- Variable-length sequences in batched data

### Sample and Variate Identification

The loss functions use sample and variate identifiers to properly aggregate losses across:
- Multiple samples in a batch
- Multiple time series (variates) in multivariate data
- Variable-length sequences

### Proper Loss Reduction

The loss functions implement proper reduction strategies to ensure:
- Fair weighting of samples regardless of sequence length
- Correct normalization based on the number of observations
- Handling of edge cases like zero observations

### Probabilistic Forecast Evaluation

The distribution losses provide proper scoring rules for evaluating probabilistic forecasts:
- Encouraging calibrated probability distributions
- Penalizing both incorrect central predictions and incorrect uncertainty estimates
- Supporting various distribution types (Normal, Student-t, Negative Binomial, etc.)

## Usage Examples

### Point Forecast Evaluation

```python
from uni2ts.loss.packed import PackedMAELoss, PackedRMSELoss

# Create loss functions
mae_loss = PackedMAELoss()
rmse_loss = PackedRMSELoss()

# Compute losses
mae = mae_loss(
    pred=predictions,
    target=targets,
    prediction_mask=pred_mask,
    observed_mask=obs_mask,
    sample_id=sample_ids,
    variate_id=variate_ids
)

rmse = rmse_loss(
    pred=predictions,
    target=targets,
    prediction_mask=pred_mask,
    observed_mask=obs_mask,
    sample_id=sample_ids,
    variate_id=variate_ids
)
```

### Probabilistic Forecast Evaluation

```python
from uni2ts.loss.packed import PackedNLLLoss

# Create loss function
nll_loss = PackedNLLLoss()

# Compute loss
nll = nll_loss(
    pred=distribution,  # A torch.distributions.Distribution object
    target=targets,
    prediction_mask=pred_mask,
    observed_mask=obs_mask,
    sample_id=sample_ids,
    variate_id=variate_ids
)
```

### Using Normalized Losses

```python
from uni2ts.loss.packed import PackedNMAELoss, PointNormType

# Create normalized loss with different normalization strategies
nmae_std = PackedNMAELoss(normalize=PointNormType.STD)  # Normalize by standard deviation
nmae_mean = PackedNMAELoss(normalize=PointNormType.MEAN)  # Normalize by mean
nmae_range = PackedNMAELoss(normalize=PointNormType.RANGE)  # Normalize by range (max - min)

# Compute normalized loss
nmae = nmae_std(
    pred=predictions,
    target=targets,
    prediction_mask=pred_mask,
    observed_mask=obs_mask,
    sample_id=sample_ids,
    variate_id=variate_ids
)
```

## Integration with the Framework

The loss module integrates with other parts of the uni2ts framework:

- **Model Module**: Loss functions are used to train models like MoiraiPretrain and MoiraiForecast
- **Evaluation Utilities**: Loss functions are used to evaluate model performance
- **Lightning Integration**: Loss functions are compatible with PyTorch Lightning's training loop

This integration allows for a seamless workflow from model training to evaluation, with consistent loss computation throughout the process.
