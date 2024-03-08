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

from contextlib import nullcontext as does_not_raise
from typing import ContextManager

import pytest
import torch
from einops import repeat

from uni2ts.module.position.additive import LearnedEmbedding, SinusoidalPositionEncoding


@pytest.mark.parametrize(
    "batch_shape, width, normalize",
    [
        (tuple(), 16, False),
        ((2,), 32, False),
        ((2, 3), 64, True),
    ],
)
@pytest.mark.parametrize(
    "seq_len, expectation",
    [
        (10, does_not_raise()),
        (100, does_not_raise()),
        (101, pytest.raises(IndexError)),
    ],
)
def test_sinusoidal_position_encoding(
    batch_shape: tuple[int, ...],
    seq_len: int,
    expectation: ContextManager,
    width: int,
    normalize: bool,
    max_len: int = 100,
):
    pe = SinusoidalPositionEncoding(width=width, max_len=max_len, normalize=normalize)

    batch_shape_str = " ".join(str(bs) for bs in batch_shape)
    pos_id = repeat(
        torch.arange(seq_len),
        f"seq_len -> {batch_shape_str} seq_len",
    )

    with expectation:
        pos = pe(pos_id)
        assert pos.shape == batch_shape + (seq_len, width)


@pytest.mark.parametrize(
    "batch_shape, width, normalize",
    [
        (tuple(), 16, False),
        ((2,), 32, False),
        ((2, 3), 64, True),
    ],
)
@pytest.mark.parametrize(
    "seq_len, expectation",
    [
        (10, does_not_raise()),
        (100, does_not_raise()),
        (101, pytest.raises(IndexError)),
    ],
)
def test_learned_embedding(
    batch_shape: tuple[int, ...],
    seq_len: int,
    expectation: ContextManager,
    width: int,
    normalize: bool,
    max_len: int = 100,
):
    pe = LearnedEmbedding(width=width, max_len=max_len)

    batch_shape_str = " ".join(str(bs) for bs in batch_shape)
    pos_id = repeat(
        torch.arange(seq_len),
        f"seq_len -> {batch_shape_str} seq_len",
    )

    with expectation:
        pos = pe(pos_id)
        assert pos.shape == batch_shape + (seq_len, width)
