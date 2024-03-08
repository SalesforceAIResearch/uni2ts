#  Copyright (c) 2024, Salesforce, Inc.
#  SPDX-License-Identifier: Apache-2
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from itertools import product
from typing import Optional

import pytest
import torch
from jaxtyping import Bool, Int

from uni2ts.common.torch_util import (
    fixed_size,
    mask_fill,
    masked_mean,
    packed_attention_mask,
    safe_div,
    sized_mean,
)


@pytest.mark.parametrize(
    "sample_id, attention_mask",
    [
        (
            torch.as_tensor(
                [
                    [1, 1, 1],
                    [1, 2, 3],
                ]
            ),
            torch.as_tensor(
                [
                    [
                        [1, 1, 1],
                        [1, 1, 1],
                        [1, 1, 1],
                    ],
                    [
                        [1, 0, 0],
                        [0, 1, 0],
                        [0, 0, 1],
                    ],
                ]
            ),
        ),
        (
            torch.as_tensor(
                [
                    [1, 1, 0],
                    [1, 0, 0],
                ]
            ),
            torch.as_tensor(
                [
                    [
                        [1, 1, 0],
                        [1, 1, 0],
                        [0, 0, 1],
                    ],
                    [
                        [1, 0, 0],
                        [0, 1, 1],
                        [0, 1, 1],
                    ],
                ]
            ),
        ),
    ],
)
def test_packed_attention_mask(
    sample_id: Int[torch.Tensor, "batch seq_len"],
    attention_mask: Bool[torch.Tensor, "batch seq_len seq_len"],
):
    assert packed_attention_mask(sample_id).eq(attention_mask).all()


@pytest.mark.parametrize(
    "batch_shape, seq_len, dim",
    [
        (tuple(), 1, 32),
        (tuple(), 10, 32),
        ((1,), 10, 128),
        ((2, 3), 20, 128),
    ],
)
@pytest.mark.parametrize("mask_ratio", [0.1, 0.5, 0.9])
def test_mask_fill(
    batch_shape: tuple[int, ...],
    seq_len: int,
    dim: int,
    mask_ratio: float,
):
    tensor = torch.randn(batch_shape + (seq_len, dim))
    mask = torch.rand(batch_shape + (seq_len,)) < mask_ratio
    value = torch.randn(dim)
    filled_tensor = mask_fill(tensor, mask, value)
    assert filled_tensor.shape == tensor.shape
    assert filled_tensor.dtype == tensor.dtype
    assert filled_tensor[mask].eq(value).all()
    assert filled_tensor[~mask].eq(tensor[~mask.to(bool)]).all()


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((1,), None),
        ((1,), (0,)),
        ((2, 3), None),
        ((2, 3), (0,)),
        ((2, 3), (0, 1)),
        ((2, 3), (-1,)),
        ((2, 3), (-1, -2)),
    ],
)
@pytest.mark.parametrize("max_size", [1, 5, 10])
@pytest.mark.parametrize("full_size", [True, False])
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_sized_mean(
    shape: tuple[int, ...],
    dim: Optional[tuple[int, ...]],
    max_size: int,
    full_size: bool,
    seed: int,
):
    torch.manual_seed(seed)
    value = torch.zeros(*shape, max_size)
    size = torch.zeros(*shape)
    orig_dim = dim

    if dim is None:
        dim = tuple(range(len(shape)))
    dim = list(dim)
    for i, d in enumerate(dim):
        if d < 0:
            dim[i] = len(shape) + d
    dim = tuple(dim)
    cum_sum = torch.zeros(tuple(x for i, x in enumerate(shape) if i not in dim))
    cum_size = torch.zeros(tuple(x for i, x in enumerate(shape) if i not in dim))
    for idx in product(*[list(range(s)) for s in shape]):
        if full_size:
            _size = torch.ones((), dtype=torch.long) * max_size
        else:
            _size = torch.randint(1, max_size + 1, ())
        _value = torch.randn(_size)
        value[idx][:_size] = _value
        size[idx] = _size

        cum_idx = tuple(x for i, x in enumerate(idx) if i not in dim)
        cum_sum[cum_idx] += _value.sum()
        cum_size[cum_idx] += _size

    mean = cum_sum / cum_size
    if full_size:
        sized_mean_value = sized_mean(value, fixed_size(value), dim=orig_dim)
    else:
        sized_mean_value = sized_mean(value, size, dim=orig_dim)
    assert torch.allclose(mean, sized_mean_value)


@pytest.mark.parametrize(
    "shape, dim",
    [
        ((1,), None),
        ((1,), (0,)),
        ((2, 3), None),
        ((2, 3), (0,)),
        ((2, 3), (0, 1)),
        ((2, 3), (-1,)),
        ((2, 3), (-1, -2)),
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_masked_mean(
    shape: tuple[int, ...],
    dim: Optional[tuple[int, ...]],
    seed: int,
):
    torch.manual_seed(seed)
    value = torch.zeros(*shape)
    mask = torch.zeros(*shape)
    orig_dim = dim

    if dim is None:
        dim = tuple(range(len(shape)))
    dim = list(dim)
    for i, d in enumerate(dim):
        if d < 0:
            dim[i] = len(shape) + d
    dim = tuple(dim)
    cum_sum = torch.zeros(tuple(x for i, x in enumerate(shape) if i not in dim))
    cum_mask = torch.zeros(tuple(x for i, x in enumerate(shape) if i not in dim))
    for idx in product(*[list(range(s)) for s in shape]):
        _mask = torch.rand(()) < 0.5
        _value = torch.randn(())

        value[idx] = _value
        mask[idx] = _mask

        cum_idx = tuple(x for i, x in enumerate(idx) if i not in dim)
        cum_sum[cum_idx] += _value * _mask
        cum_mask[cum_idx] += _mask

    mean = safe_div(cum_sum, cum_mask)
    masked_mean_value = masked_mean(value, mask, dim=orig_dim)
    assert torch.allclose(mean, masked_mean_value)
