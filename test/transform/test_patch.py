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
from typing import ContextManager, Optional

import numpy as np
import pytest

from uni2ts.transform import PackFields, PatchCrop
from uni2ts.transform.patch import (
    DefaultPatchSizeConstraints,
    FixedPatchSizeConstraints,
    GetPatchSize,
    Patchify,
    PatchSizeConstraints,
)


@pytest.mark.parametrize(
    "start, stop",
    [
        (10, None),
        (10, 20),
    ],
)
def fixed_patch_size_constraints(start: int, stop: Optional[int]):
    constraints = FixedPatchSizeConstraints(start, stop)
    for freq in ["H", "W", "T"]:
        freq_constraint = constraints(freq)

        assert isinstance(freq_constraint, range)
        assert min(freq_constraint) == start

        if stop is not None:
            assert max(freq_constraint) == stop
        else:
            assert max(freq_constraint) == start


@pytest.mark.parametrize(
    "freq", ["30S", "60S", "T", "5T", "30T", "H", "6H", "24H", "D", "7D", "W", "M", "Q"]
)
def test_default_patch_size_constraints(freq: str):
    freq_constraint = DefaultPatchSizeConstraints()(freq)
    start, stop = DefaultPatchSizeConstraints.DEFAULT_RANGES[freq[-1]]
    assert min(freq_constraint) == start
    assert max(freq_constraint) == stop


@pytest.mark.parametrize(
    "freq", ["30S", "60S", "T", "5T", "30T", "H", "6H", "24H", "D", "7D", "W", "M", "Q"]
)
@pytest.mark.parametrize(
    "patch_sizes",
    [
        (8, 16, 32, 64, 128),
        range(8, 129),
    ],
)
@pytest.mark.parametrize(
    "patch_size_constraints, expectation",
    [
        (DefaultPatchSizeConstraints(), does_not_raise()),
        (
            FixedPatchSizeConstraints(1),
            pytest.raises(
                AssertionError, match=r"no valid patch size candidates for \w+"
            ),
        ),
    ],
)
def test_get_patch_size(
    create_data_entry,
    freq: str,
    patch_sizes: tuple[int, ...] | range,
    patch_size_constraints: PatchSizeConstraints,
    expectation: ContextManager,
):
    with expectation:
        np.random.seed(0)
        data_entry = create_data_entry(1000, freq, 1)
        transform = GetPatchSize(
            min_time_patches=2,
            patch_sizes=patch_sizes,
            patch_size_constraints=patch_size_constraints,
        )

        transformed_data_entry = transform(data_entry.copy())

        assert "patch_size" in transformed_data_entry
        assert isinstance(transformed_data_entry["patch_size"], (np.int64, int))
        assert transformed_data_entry["patch_size"] in transform.patch_sizes
        assert transformed_data_entry["patch_size"] in transform.patch_size_constraints(
            freq
        )


@pytest.mark.parametrize("target_dim", [1, 2, 3])
def test_patchify(
    create_data_entry,
    target_dim: int,
):
    max_patch_size = 128
    np.random.seed(0)
    fields = ("target",)

    data_entry = create_data_entry(
        1000,
        "H",
        target_dim=target_dim,
    )

    data_entry = (
        GetPatchSize(
            min_time_patches=2,
        )
        + PatchCrop(2, 512, fields=fields, will_flatten=False)
        + PackFields("target", fields=fields)
    )(data_entry)
    transformed_data_entry = Patchify(max_patch_size, fields=fields)(data_entry.copy())

    target = data_entry["target"]
    transformed_target = transformed_data_entry["target"]
    patch_size = transformed_data_entry["patch_size"]

    assert transformed_target.ndim == target.ndim + 1
    assert transformed_target.shape[-2] == target.shape[-1] // patch_size
    assert transformed_target.shape[-1] == max_patch_size
    assert np.all(
        transformed_target[:, patch_size:]
        == np.zeros_like(transformed_target[:, patch_size:])
    )
    assert transformed_target.shape[0] == target_dim
