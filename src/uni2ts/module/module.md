# Module

The `module` directory contains neural network building blocks used throughout the uni2ts framework. These modules are designed to handle time series data efficiently, with a focus on transformer-based architectures.

## Core Components

### Attention Mechanisms

The `attention.py` file implements various attention mechanisms:

- **GroupedQueryAttention**: A flexible attention mechanism that supports grouped queries, where multiple query heads can attend to the same key-value pairs. This is a generalization of both multi-head and multi-query attention.
- **MultiQueryAttention**: An attention mechanism where all query heads share the same key-value projections, reducing computational and memory requirements.
- **MultiHeadAttention**: The standard multi-head attention mechanism where each head has its own key, query, and value projections.

These attention mechanisms support:
- Attention biases for incorporating positional information
- Query-key projections for rotary position embeddings
- Separate handling of time and variate dimensions

### Feed-Forward Networks

The `ffn.py` file implements various feed-forward network architectures:

- **FeedForward**: A standard feed-forward network with two linear layers and an activation function.
- **GatedLinearUnitFeedForward**: A feed-forward network with a gating mechanism, similar to the one used in LLaMA and other modern transformer architectures.
- **MoEFeedForward**: A Mixture of Experts feed-forward network that routes tokens to different expert networks based on their content.

### Transformer Architecture

The `transformer.py` file implements the transformer encoder architecture:

- **TransformerEncoderLayer**: A single layer of the transformer encoder, combining self-attention and feed-forward networks with residual connections and normalization.
- **TransformerEncoder**: A stack of transformer encoder layers with shared or separate positional encodings.

The transformer implementation supports:
- Pre-norm or post-norm configurations
- Mixture of Experts layers
- Customizable attention and feed-forward components

### Time Series Embeddings

The `ts_embed.py` file implements embeddings specific to time series data:

- **MultiInSizeLinear**: A linear layer that handles inputs of different sizes, useful for processing patches of different lengths.
- **MultiOutSizeLinear**: A linear layer that produces outputs of different sizes, useful for generating predictions of different lengths.
- **FeatLinear**: A linear layer that applies different transformations based on feature size.

### Normalization

The `norm.py` file implements normalization techniques:

- **RMSNorm**: Root Mean Square Normalization, a variant of Layer Normalization that normalizes by the RMS (root mean square) of the activations.

### Packed Sequence Scaling

The `packed_scaler.py` file implements scaling mechanisms for packed sequences:

- **PackedScaler**: Base class for scalers that handle packed sequences.
- **PackedNOPScaler**: A no-operation scaler that doesn't modify the data.
- **PackedStdScaler**: A scaler that standardizes the data by subtracting the mean and dividing by the standard deviation.
- **PackedAbsMeanScaler**: A scaler that scales the data by the mean of absolute values.

### Positional Encodings

The `position/` directory contains various positional encoding mechanisms:

- **Additive Encodings** (`additive.py`):
  - **LearnedEmbedding**: Learnable position embeddings.
  - **SinusoidalPositionEncoding**: Fixed sinusoidal position encodings.

- **Attention Biases** (`attn_bias.py`):
  - **AttentionBias**: Base class for attention biases.
  - **BinaryAttentionBias**: Binary attention bias for masking.
  - **LinearAttentionBias**: Linear attention bias based on position differences.
  - **RelativeAttentionBias**: Relative position attention bias.

- **Attention Projections** (`attn_projection.py`):
  - **Projection**: Base class for projections.
  - **IdentityProjection**: Identity projection that doesn't modify the input.
  - **LearnedProjection**: Learnable projection.
  - **RotaryProjection**: Rotary position embedding (RoPE).
  - **QueryKeyProjection**: Wrapper for applying projections to queries and keys.

## Key Features

### Handling Variable-Length Sequences

Many modules are designed to handle variable-length sequences through:
- Packed sequence representations
- Masking mechanisms
- Specialized linear layers for different input/output sizes

### Positional Information

The framework provides multiple ways to incorporate positional information:
- Additive position embeddings
- Attention biases based on relative positions
- Rotary position embeddings

### Mixture of Experts

The framework supports Mixture of Experts (MoE) architectures:
- Token routing based on content
- Multiple expert feed-forward networks
- Efficient computation through sparse activation

### Efficient Attention

The attention mechanisms are designed for efficiency:
- Grouped query attention reduces computation
- Query-key normalization improves stability
- Support for various attention patterns through biases

## Usage Examples

### Creating a Transformer Encoder

```python
encoder = TransformerEncoder(
    d_model=256,
    num_layers=4,
    num_heads=8,
    pre_norm=True,
    attn_dropout_p=0.1,
    dropout_p=0.1,
    norm_layer=RMSNorm,
    activation=F.silu,
    use_glu=True,
    use_qk_norm=True,
    var_attn_bias_layer=partial(BinaryAttentionBias),
    time_qk_proj_layer=partial(
        QueryKeyProjection,
        proj_layer=RotaryProjection,
        kwargs=dict(max_len=1024),
    ),
)
```

### Using MultiInSizeLinear for Variable Patch Sizes

```python
in_proj = MultiInSizeLinear(
    in_features_ls=(16, 32, 64),  # Support patches of size 16, 32, and 64
    out_features=256,  # Project to hidden dimension of 256
)

# For a batch with different patch sizes
output = in_proj(input_tensor, patch_size)
```

### Applying Scaling to Packed Sequences

```python
scaler = PackedStdScaler()
loc, scale = scaler(
    target,
    observed_mask,
    sample_id,
    variate_id,
)
scaled_target = (target - loc) / scale
