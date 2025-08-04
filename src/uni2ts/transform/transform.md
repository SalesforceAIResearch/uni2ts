# Transform Module

The `transform` module provides a collection of data transformation utilities for time series data processing in the uni2ts framework. These transformations are designed to prepare time series data for model training, evaluation, and inference.

## Core Components

### Base Transformations

The module is built around the `Transformation` abstract base class, which defines the interface for all transformations:

- **Transformation**: Abstract base class for all transformations with a `__call__` method that takes a data entry dictionary and returns a transformed dictionary.
- **Chain**: Combines multiple transformations into a single transformation that applies them sequentially.
- **Identity**: A no-op transformation that returns the input unchanged.

### Data Manipulation Categories

The transformations are organized into several categories based on their functionality:

#### Cropping and Patching

- **PatchCrop**: Crops fields in a data entry in the temporal dimension based on a patch size.
- **EvalCrop**: Crops fields for evaluation, with specific handling for context and prediction windows.
- **GetPatchSize**: Determines an appropriate patch size for a time series based on frequency and constraints.
- **Patchify**: Converts time series data into patches of a specified size.
- **PatchSizeConstraints**: Abstract class for defining constraints on patch sizes.
- **FixedPatchSizeConstraints**: Implements fixed constraints on patch sizes.
- **DefaultPatchSizeConstraints**: Provides default patch size ranges based on time series frequency.

#### Task-Specific Transformations

- **MaskedPrediction**: Creates a mask for prediction tasks, typically masking the end of the time series.
- **ExtendMask**: Extends a mask to cover additional fields.
- **EvalMaskedPrediction**: Creates a mask for evaluation, with a fixed mask length.

#### Reshaping and Packing

- **SequencifyField**: Repeats a field to match the sequence length of another field.
- **PackFields**: Combines multiple fields into a single packed field.
- **FlatPackFields**: Similar to PackFields but flattens the time dimension.
- **PackCollection**: Packs a collection of arrays into a single array.
- **FlatPackCollection**: Similar to PackCollection but flattens the time dimension.
- **Transpose**: Transposes the axes of specified fields.

#### Field Manipulation

- **SelectFields**: Selects specific fields from a data entry.
- **RemoveFields**: Removes specific fields from a data entry.
- **SetValue**: Sets a field to a specified value.
- **LambdaSetFieldIfNotPresent**: Sets a field using a lambda function if it's not already present.

#### Feature Engineering

- **AddObservedMask**: Adds an observed mask to indicate which values are observed.
- **AddTimeIndex**: Adds a time index field.
- **AddVariateIndex**: Adds a variate index field.

#### Padding and Imputation

- **Pad**: Pads fields to a specified length.
- **PadFreq**: Pads fields based on frequency.
- **EvalPad**: Pads fields for evaluation.
- **ImputeTimeSeries**: Base class for time series imputation.
- **LastValueImputation**: Imputes missing values with the last observed value.
- **DummyValueImputation**: Imputes missing values with a dummy value.

#### Sampling

- **SampleDimension**: Samples from a specific dimension of the data.

## Key Features

### Composability

Transformations can be composed using the `Chain` class or the `+` operator, allowing for complex data processing pipelines to be built from simple building blocks.

```python
transform = AddTimeIndex() + AddObservedMask() + MaskedPrediction(min_mask_ratio=0.1, max_mask_ratio=0.2)
```

### Flexibility

The transformations are designed to work with a variety of data formats and structures, including:
- Lists of arrays
- Dictionaries of arrays
- Numpy arrays
- Nested structures

### Task-Specific Processing

The module includes transformations specifically designed for common time series tasks:
- Forecasting: Masking the end of the time series for prediction
- Imputation: Filling in missing values
- Feature engineering: Adding time and variate indices

### Patching for Efficient Processing

The patching transformations enable efficient processing of long time series by breaking them into smaller, manageable patches. This is particularly useful for transformer-based models that have quadratic complexity with respect to sequence length.

## Usage Examples

### Data Preparation for Training

```python
from uni2ts.transform import Chain, GetPatchSize, PatchCrop, Patchify, AddTimeIndex, MaskedPrediction

# Create a transformation pipeline for training
transform = Chain([
    # Determine patch size based on time series frequency
    GetPatchSize(min_time_patches=4),
    
    # Crop the time series to a random segment
    PatchCrop(min_time_patches=4, max_patches=16, will_flatten=True),
    
    # Convert to patches
    Patchify(max_patch_size=128),
    
    # Add time indices for positional encoding
    AddTimeIndex(),
    
    # Mask the end of the time series for prediction
    MaskedPrediction(min_mask_ratio=0.1, max_mask_ratio=0.2)
])

# Apply the transformation to a data entry
transformed_data = transform(data_entry)
```

### Data Preparation for Evaluation

```python
from uni2ts.transform import Chain, EvalCrop, EvalPad, AddTimeIndex, EvalMaskedPrediction

# Create a transformation pipeline for evaluation
transform = Chain([
    # Crop the time series to the evaluation window
    EvalCrop(
        offset=0,
        distance=1,
        prediction_length=24,
        context_length=72,
        fields=("target", "observed_mask")
    ),
    
    # Pad if necessary
    EvalPad(
        pad_length=96,  # context_length + prediction_length
        fields=("target", "observed_mask")
    ),
    
    # Add time indices for positional encoding
    AddTimeIndex(),
    
    # Mask the prediction window
    EvalMaskedPrediction(
        mask_length=24,  # prediction_length
        target_field="target",
        truncate_fields=("target",)
    )
])

# Apply the transformation to a data entry
transformed_data = transform(data_entry)
```

### Feature Engineering

```python
from uni2ts.transform import Chain, AddObservedMask, AddVariateIndex, AddTimeIndex

# Create a transformation pipeline for feature engineering
transform = Chain([
    # Add an observed mask based on NaN values
    AddObservedMask(),
    
    # Add variate indices for variable-specific processing
    AddVariateIndex(),
    
    # Add time indices for temporal processing
    AddTimeIndex()
])

# Apply the transformation to a data entry
transformed_data = transform(data_entry)
```

## Integration with the Framework

The transformations in this module are used throughout the uni2ts framework, particularly in:

- The `Dataset` class for data loading and preprocessing
- The `MoiraiPretrain` class for pretraining Moirai models
- The `MoiraiFinetune` class for finetuning Moirai models
- The evaluation utilities for model evaluation

These classes use the transformations to prepare data for model training, evaluation, and inference, ensuring consistent data processing across the framework.
