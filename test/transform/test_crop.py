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

import numpy as np
import pytest
from einops import rearrange

from uni2ts.transform import GetPatchSize, PackFields
from uni2ts.transform.crop import EvalCrop, PatchCrop


def test_min_max():
    with pytest.raises(AssertionError):
        PatchCrop(32, 31)


def array_in(arr: np.ndarray, test_arr: np.ndarray):
    arr = arr.T
    test_arr = test_arr.T
    print(arr.shape, test_arr.shape)
    ax = (1,) if arr.ndim == 1 else (1, 2)
    return (
        np.isclose(
            arr[
                np.arange(len(arr) - len(test_arr) + 1)[:, None]
                + np.arange(len(test_arr))
            ],
            test_arr,
        )
        .all(axis=ax)
        .any(axis=0)
    )


@pytest.mark.parametrize(
    "arr, test_arr",
    [
        (
            np.arange(10, dtype=float),
            np.asarray([-1.0, -1.0]),
        ),
        (
            np.asarray([[0.0, 1.0], [1.0, 2.0]]),
            np.asarray([[-1.0], [1.0]]),
        ),
    ],
)
def test_not_in_array(arr: np.ndarray, test_arr: np.ndarray):
    with pytest.raises(AssertionError):
        assert array_in(arr, test_arr)


@pytest.mark.parametrize(
    "min_time_patches, max_patches, flatten, offset, length, dim, patch_size, expectation",
    [
        (5, 10, False, False, 5, 1, 1, does_not_raise()),
        (
            5,
            10,
            False,
            False,
            4,
            1,
            1,
            pytest.raises(ValueError, match=r"max_patches=\d+ < min_time_patches=\d+"),
        ),
        (5, 10, False, False, 10, 1, 2, does_not_raise()),
        (
            5,
            10,
            False,
            False,
            9,
            1,
            2,
            pytest.raises(ValueError, match=r"max_patches=\d+ < min_time_patches=\d+"),
        ),
        (5, 10, True, False, 5, 2, 1, does_not_raise()),
        (
            5,
            10,
            True,
            False,
            4,
            2,
            1,
            pytest.raises(ValueError, match=r"max_patches=\d+ < min_time_patches=\d+"),
        ),
        (5, 10, True, False, 10, 2, 2, does_not_raise()),
        (
            5,
            10,
            True,
            False,
            9,
            2,
            2,
            pytest.raises(ValueError, match=r"max_patches=\d+ < min_time_patches=\d+"),
        ),
        (5, 10, False, True, 5, 1, 1, does_not_raise()),
        (5, 10, False, True, 11, 1, 2, does_not_raise()),
        (5, 10, True, True, 5, 2, 1, does_not_raise()),
        (5, 10, True, True, 11, 2, 2, does_not_raise()),
    ],
)
@pytest.mark.parametrize("seed", [i for i in range(5)])
def test_patch_crop(
    create_data_entry,
    min_time_patches: int,
    max_patches: int,
    flatten: bool,
    offset: bool,
    length: bool,
    dim: int,
    patch_size: int,
    expectation: ContextManager,
    seed: int,
):
    with expectation:
        data_entry = create_data_entry(
            length=length,
            freq="H",
            target_dim=dim,
        ) | dict(patch_size=patch_size)
        np.random.seed(seed)
        patch_crop = PatchCrop(
            min_time_patches,
            max_patches,
            will_flatten=flatten,
            offset=offset,
            fields=("target",),
        )
        transformed = patch_crop(data_entry)

        if flatten:
            out = rearrange(
                transformed["target"],
                "... (time patch) -> (... time) patch",
                patch=transformed["patch_size"],
            )
            assert out.shape[0] // dim >= min_time_patches
            assert out.shape[0] // dim <= max_patches
        else:
            out = rearrange(
                transformed["target"],
                "... (time patch) -> ... time patch",
                patch=transformed["patch_size"],
            )
            assert out.shape[-2] >= min_time_patches
            assert out.shape[-2] <= max_patches


@pytest.mark.parametrize(
    "length, freq",
    [(1000, "T")]
    + [(l, "H") for l in [1000, 2000, 3000]]
    + [(100, f) for f in ["B", "W", "7D", "D", "15T"]],
)
@pytest.mark.parametrize("target_dim", [1, 3])
@pytest.mark.parametrize("past_feat_dynamic_real_dim", [None, 1, 3])
@pytest.mark.parametrize("flatten", [True, False])
@pytest.mark.parametrize("seed", [i for i in range(3)])
def test_patch_crop_transform_pipeline(
    create_data_entry,
    length: int,
    freq: str,
    target_dim: int,
    past_feat_dynamic_real_dim: int,
    flatten: bool,
    seed: int,
):
    min_patches = 2
    max_patches = 512

    np.random.seed(seed)
    data_entry = create_data_entry(
        length=length,
        freq=freq,
        target_dim=target_dim,
        past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
    )

    transformed_data_entry = (
        GetPatchSize(min_patches)
        + PatchCrop(
            min_patches,
            max_patches,
            will_flatten=flatten,
            fields=("target",),
            optional_fields=("past_feat_dynamic_real",),
        )
        + PackFields("target", fields=("target",))
        + PackFields(
            "past_feat_dynamic_real",
            fields=tuple(),
            optional_fields=("past_feat_dynamic_real",),
        )
    )(data_entry.copy())

    target = np.asarray(data_entry["target"])
    transformed_target = transformed_data_entry["target"]
    transformed_target_length = transformed_target.shape[-1]
    patch_size = transformed_data_entry["patch_size"]

    assert transformed_target_length <= length
    assert transformed_target_length % patch_size == 0
    assert array_in(target, transformed_target)
    assert target.ndim == transformed_target.ndim
    if target_dim > 1:
        assert target_dim == transformed_target.shape[0]
    if flatten:
        assert (
            transformed_target_length
            // patch_size
            * (target_dim + (past_feat_dynamic_real_dim or 0))
            >= min_patches
        )
        assert (
            transformed_target_length
            // patch_size
            * (target_dim + (past_feat_dynamic_real_dim or 0))
            <= max_patches
        )
    else:
        assert transformed_target_length // patch_size >= min_patches
        assert transformed_target_length // patch_size <= max_patches
    if past_feat_dynamic_real_dim is not None:
        feat = np.asarray(data_entry["past_feat_dynamic_real"])
        transformed_feat = transformed_data_entry["past_feat_dynamic_real"]
        assert transformed_target_length == transformed_feat.shape[-1]
        assert array_in(feat, transformed_feat)
        assert feat.ndim == transformed_feat.ndim


@pytest.mark.parametrize("dim", [1, 3])
@pytest.mark.parametrize(
    "length, prediction_length, context_length, windows, distance, offset, expectation",
    [
        (100, 10, 1, 1, 1, -10, does_not_raise()),
        (100, 2, 1, 5, 1, -10, does_not_raise()),
        (100, 2, 1, 5, 2, -10, does_not_raise()),
        (100, 2, 1, 5, 2, -50, does_not_raise()),
        (100, 10, 1, 1, 1, 50, does_not_raise()),
        (1000, 10, 1, 1, 1, -10, does_not_raise()),
        (1000, 2, 1, 5, 1, -10, does_not_raise()),
        (1000, 2, 1, 5, 2, -10, does_not_raise()),
        (1000, 2, 1, 5, 2, -50, does_not_raise()),
        (1000, 10, 1, 1, 1, 50, does_not_raise()),
        (100, 10, 1, 3, 10, -20, pytest.raises(AssertionError)),
        (20, 10, 1, 1, 1, -30, pytest.raises(AssertionError)),
    ],
)
def test_eval_crop(
    create_data_entry,
    length: int,
    dim: int,
    prediction_length: int,
    context_length: int,
    windows: int,
    distance: int,
    offset: int,
    expectation: ContextManager,
):
    with expectation:
        for window in range(windows):
            # data_entry = {
            #     "target": np.random.randn(*((dim, length) if dim > 1 else (length,))),
            #     "window": window,
            # }
            data_entry = create_data_entry(
                length=length,
                target_dim=dim,
                freq="H",
            ) | dict(window=window)
            transform = EvalCrop(
                offset,
                distance,
                prediction_length,
                context_length,
                fields=("target",),
            )
            transformed_data_entry = transform(data_entry.copy())

            fcst_start = offset + window * distance
            a = fcst_start - context_length
            b = fcst_start + prediction_length

            assert transformed_data_entry["target"][0].shape[0] == b - a
            assert np.all(
                np.asarray(transformed_data_entry["target"])
                == np.asarray(data_entry["target"])[..., a : b or None]
            )

            if dim > 1:
                assert len(transformed_data_entry["target"]) == len(
                    data_entry["target"]
                )
