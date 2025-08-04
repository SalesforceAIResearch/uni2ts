# Optim Module

The `optim` module provides optimization utilities for training models in the uni2ts framework, with a focus on learning rate scheduling.

## Overview

The module primarily consists of learning rate schedulers that control how the learning rate changes during training. These schedulers are essential for effective training of deep learning models, as they help with convergence, avoiding local minima, and improving generalization.

## Key Components

### SchedulerType Enum

The `SchedulerType` enum defines the available learning rate scheduler types:

- `LINEAR`: Linear decay with warmup
- `COSINE`: Cosine decay with warmup
- `COSINE_WITH_RESTARTS`: Cosine decay with hard restarts and warmup
- `POLYNOMIAL`: Polynomial decay with warmup
- `CONSTANT`: Constant learning rate
- `CONSTANT_WITH_WARMUP`: Constant learning rate after warmup
- `INVERSE_SQRT`: Inverse square root decay with warmup
- `REDUCE_ON_PLATEAU`: Reduce learning rate when a metric plateaus

### get_scheduler Function

The `get_scheduler` function provides a unified API to get any scheduler from its name. It takes the following parameters:

- `name`: The name of the scheduler to use (string or SchedulerType)
- `optimizer`: The PyTorch optimizer to apply the scheduler to
- `num_warmup_steps`: The number of warmup steps (optional, required by most schedulers)
- `num_training_steps`: The total number of training steps (optional, required by most schedulers)
- `scheduler_specific_kwargs`: Extra parameters for specific schedulers

## Scheduler Implementations

### Constant Schedulers

- `get_constant_schedule`: Creates a schedule with a constant learning rate
- `get_constant_schedule_with_warmup`: Creates a schedule with a constant learning rate after a warmup period

### Linear Schedulers

- `get_linear_schedule_with_warmup`: Creates a schedule with a learning rate that decreases linearly from the initial value to 0, after a warmup period

### Cosine Schedulers

- `get_cosine_schedule_with_warmup`: Creates a schedule with a learning rate that follows a cosine decay from the initial value to 0, after a warmup period
- `get_cosine_with_hard_restarts_schedule_with_warmup`: Similar to cosine decay but with hard restarts at specified intervals

### Other Schedulers

- `get_polynomial_decay_schedule_with_warmup`: Creates a schedule with a learning rate that decreases as a polynomial decay
- `get_inverse_sqrt_schedule`: Creates a schedule with an inverse square-root learning rate decay
- `get_reduce_on_plateau_schedule`: Creates a schedule that reduces the learning rate when a metric has stopped improving

## Warmup Mechanism

Most schedulers in this module support a warmup phase, where the learning rate gradually increases from 0 to the initial value over a specified number of steps. This helps stabilize training in the early stages, particularly for transformer-based models.

## Usage Examples

### Basic Usage

```python
from uni2ts.optim import SchedulerType, get_scheduler
import torch

# Create an optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

# Create a scheduler
scheduler = get_scheduler(
    name=SchedulerType.COSINE,
    optimizer=optimizer,
    num_warmup_steps=1000,
    num_training_steps=10000
)

# Use in training loop
for epoch in range(num_epochs):
    for batch in dataloader:
        # Forward pass, loss calculation, backward pass
        optimizer.step()
        scheduler.step()  # Update learning rate
```

### Using with Lightning

```python
def configure_optimizers(self):
    optimizer = torch.optim.AdamW(
        self.parameters(),
        lr=self.learning_rate,
        weight_decay=0.01
    )
    scheduler = get_scheduler(
        name=SchedulerType.COSINE_WITH_RESTARTS,
        optimizer=optimizer,
        num_warmup_steps=self.num_warmup_steps,
        num_training_steps=self.num_training_steps,
        scheduler_specific_kwargs={"num_cycles": 3}
    )
    return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": scheduler,
            "interval": "step"
        }
    }
```

## Integration with the Framework

The schedulers in this module are used throughout the uni2ts framework, particularly in:

- The `MoiraiPretrain` class for pretraining Moirai models
- The `MoiraiFinetune` class for finetuning Moirai models
- The `MoiraiLinearProbe` class for linear probing

These classes use the `configure_optimizers` method to set up the optimizer and scheduler, typically using the cosine scheduler with restarts for effective training.
