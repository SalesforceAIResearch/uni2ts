# Common Module

The `common` module provides a collection of utility functions and classes that are used throughout the `uni2ts` project. It includes tools for handling environment variables, creating abstract class properties, defining custom types, and working with PyTorch and Hydra.

## Files

### `core.py`

-   **`abstract_class_property`**: A class decorator that ensures specified class-level properties are implemented by subclasses. This is useful for creating abstract base classes with required class-level attributes.

### `env.py`

-   **`Env`**: A singleton class that manages environment variables for the project. It loads variables from a `.env` file and provides them as class attributes, automatically converting path-related variables to `Path` objects.
-   **`get_path_var`**: A helper function that retrieves an environment variable and converts it to a `Path` object.

### `hydra_util.py`

-   **`register_resolver`**: A decorator for registering new resolvers with OmegaConf.
-   **Custom Resolvers**:
    -   `resolve_as_tuple`: Resolves a list from a Hydra config as a tuple.
    -   `resolve_cls_getattr`: Gets an attribute from a class specified by name.
    -   `resolve_floordiv`: Performs floor division.
    -   `resolve_mul`: Performs multiplication.

### `sampler.py`

-   **`Sampler`**: A type alias for a sampler function.
-   **Sampler Functions**:
    -   `uniform_sampler`: Samples a random integer uniformly from [1, n].
    -   `binomial_sampler`: Samples a random integer from a binomial distribution.
    -   `beta_binomial_sampler`: Samples a random integer from a beta-binomial distribution.
-   **`get_sampler`**: A factory function that returns a sampler function based on a string identifier.

### `torch_util.py`

-   **`numpy_to_torch_dtype_dict`**: A dictionary mapping numpy dtypes to their PyTorch counterparts.
-   **Attention Mask Functions**:
    -   `packed_attention_mask`: Creates a packed attention mask for elements with the same sample ID.
    -   `packed_causal_attention_mask`: Creates a packed causal attention mask.
-   **Tensor Manipulation Functions**:
    -   `mask_fill`: Fills elements of a tensor with a given value where a mask is True.
    -   `safe_div`: Performs division, avoiding NaNs by replacing zeros in the denominator with 1.
    -   `size_to_mask`: Converts a tensor of sizes to a boolean mask.
    -   `fixed_size`: Returns a tensor of sizes, where each size is the last dimension of the input tensor.
    -   `sized_mean`: Computes the mean of a tensor with variable-sized sequences.
    -   `masked_mean`: Computes the mean of a tensor, ignoring masked values.
    -   `unsqueeze_trailing_dims`: Unsqueezes a tensor to match the number of dimensions of a target shape.

### `typing.py`

-   **Custom `jaxtyping` Dtypes**:
    -   `DateTime64`: For `numpy.datetime64`.
    -   `Character`: For `numpy.str_`.
-   **Type Aliases**: Defines a set of type aliases for data structures used throughout the project, including types for data preparation, indexing, and data loading.
