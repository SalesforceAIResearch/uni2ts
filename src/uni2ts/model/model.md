# Model Module

The `model` module contains the implementation of the Moirai time series forecasting model and its variants. Moirai is a transformer-based model designed for probabilistic time series forecasting with a focus on handling multivariate time series data.

## Core Components

### MoiraiModule

The `MoiraiModule` class is the core neural network implementation of the Moirai model. It inherits from `nn.Module` and `PyTorchModelHubMixin` to support loading from the HuggingFace Hub.

Key features of the MoiraiModule:
- Transformer-based architecture for time series modeling
- Support for variable patch sizes for efficient processing of long sequences
- Probabilistic output through distribution objects
- Scaling mechanism for input normalization
- Integration with HuggingFace Hub for model sharing

The forward pass of MoiraiModule follows these steps:
1. Apply scaling to observations
2. Project from observations to representations
3. Replace prediction window with learnable mask
4. Apply transformer layers
5. Project from representations to distribution parameters
6. Return distribution object

### MoiraiPretrain

The `MoiraiPretrain` class is a PyTorch Lightning module that implements the pretraining process for Moirai models. It handles:

- Data preprocessing through transformations
- Training and validation steps
- Optimization configuration
- Logging of metrics

The pretraining process uses a masked prediction task, where parts of the time series are masked and the model is trained to predict the masked values.

### MoiraiForecast

The `MoiraiForecast` class is a PyTorch Lightning module that implements the forecasting functionality for Moirai models. It provides:

- Integration with GluonTS for standardized forecasting
- Support for different patch sizes
- Context management for prediction parameters
- Conversion between different data formats
- Evaluation of prediction quality

The forecasting process can automatically select the best patch size for a given time series, improving prediction accuracy.

### MoiraiMoE

The `moirai_moe` submodule contains an extension of the Moirai model that incorporates Mixture of Experts (MoE) layers. This variant uses:

- Sparse activation of expert networks
- Content-based routing of tokens
- Improved parameter efficiency through conditional computation

## Key Features

### Patching Mechanism

Moirai uses a patching mechanism to efficiently process long time series:

- Time series are divided into patches of fixed size
- Patches are processed independently by the transformer
- Different patch sizes can be used for different time series
- The model can automatically select the best patch size

### Probabilistic Forecasting

Moirai produces probabilistic forecasts through distribution objects:

- Various distribution types are supported (Normal, Student-t, Negative Binomial, etc.)
- Distribution parameters are learned by the model
- Samples can be drawn from the distribution for Monte Carlo estimation
- Uncertainty quantification is built into the model

### Multivariate Time Series Support

Moirai is designed to handle multivariate time series:

- Each variate can have its own characteristics
- Attention mechanisms capture dependencies between variates
- Scaling is applied per variate
- Predictions are made for all variates simultaneously

### Flexible Input Features

The model supports various types of input features:

- Target time series (the main variable to forecast)
- Dynamic real features (time-varying covariates)
- Past dynamic real features (historical covariates)
- Each feature type can have its own dimensionality

## Usage Examples

### Pretraining a Moirai Model

```python
from uni2ts.model.moirai import MoiraiPretrain
from uni2ts.distribution import StudentTOutput
import lightning as L

# Create a MoiraiPretrain instance
model = MoiraiPretrain(
    min_patches=4,
    min_mask_ratio=0.1,
    max_mask_ratio=0.2,
    max_dim=100,
    num_training_steps=10000,
    num_warmup_steps=1000,
    module_kwargs={
        "distr_output": StudentTOutput(),
        "d_model": 256,
        "num_layers": 4,
        "patch_sizes": (16, 32, 64),
        "max_seq_len": 256,
        "attn_dropout_p": 0.1,
        "dropout_p": 0.1,
        "scaling": True,
    }
)

# Train the model
trainer = L.Trainer(max_steps=10000, devices=4, strategy="ddp")
trainer.fit(model, train_dataloader, val_dataloader)
```

### Forecasting with a Pretrained Model

```python
from uni2ts.model.moirai import MoiraiForecast
import torch

# Load a pretrained model
forecast_model = MoiraiForecast(
    prediction_length=24,
    target_dim=5,
    feat_dynamic_real_dim=3,
    past_feat_dynamic_real_dim=0,
    context_length=72,
    module=pretrained_module,
    patch_size="auto",  # Automatically select the best patch size
    num_samples=100,
)

# Create a predictor
predictor = forecast_model.create_predictor(batch_size=32, device="cuda")

# Make predictions
predictions = predictor.predict(dataset)
```

### Using the Model with GluonTS

```python
from gluonts.dataset.repository.datasets import get_dataset
from gluonts.evaluation import make_evaluation_predictions, Evaluator

# Get a dataset
dataset = get_dataset("electricity")

# Create a predictor
predictor = forecast_model.create_predictor(batch_size=32)

# Make predictions
forecast_it, ts_it = make_evaluation_predictions(
    dataset=dataset.test,
    predictor=predictor,
    num_samples=100,
)

# Evaluate predictions
evaluator = Evaluator()
metrics = evaluator(ts_it, forecast_it)
```

## Integration with the Framework

The model module integrates with other parts of the uni2ts framework:

- **Distribution Module**: Provides the probabilistic output distributions
- **Module Module**: Provides the neural network building blocks
- **Transform Module**: Provides data preprocessing transformations
- **Loss Module**: Provides loss functions for training
- **Optim Module**: Provides optimization utilities

This integration allows for a seamless workflow from data preprocessing to model training and evaluation.
