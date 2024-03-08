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

from typing import Optional

import numpy as np
import pytest

from uni2ts.transform.task import EvalMaskedPrediction, ExtendMask, MaskedPrediction


@pytest.mark.parametrize("length, target_dim", [(5, 1), (10, 2)])
@pytest.mark.parametrize(
    "min_mask_ratio, max_mask_ratio",
    [
        (0.1, 0.2),
        (0.2, 0.3),
        (0.1, 0.9),
    ],
)
@pytest.mark.parametrize(
    "truncate_map",
    [
        dict(another=1),
        dict(another1=2, another2=3),
    ],
)
@pytest.mark.parametrize(
    "optional_truncate_map",
    [
        dict(),
        dict(optional=2),
    ],
)
@pytest.mark.parametrize("patch_size", [None, 2])
@pytest.mark.parametrize("truncate_list", [False, True])
def test_masked_prediction(
    length: int,
    target_dim: int,
    min_mask_ratio: float,
    max_mask_ratio: float,
    truncate_map: dict[str, int],
    optional_truncate_map: dict[str, int],
    patch_size: Optional[int],
    truncate_list: bool,
):
    patch_dim = () if patch_size is None else (patch_size,)
    data_entry = {"target": np.random.randn(*((target_dim, length) + patch_dim))}
    for field, dim in (truncate_map | optional_truncate_map).items():
        if truncate_list:
            data_entry[field] = [
                np.random.randn(*((dim, length) + patch_dim)) for _ in range(2)
            ]
        else:
            data_entry[field] = np.random.randn(*((dim, length) + patch_dim))
    masked_prediction = MaskedPrediction(
        min_mask_ratio,
        max_mask_ratio,
        target_field="target",
        truncate_fields=tuple(truncate_map.keys()),
        optional_truncate_fields=tuple(optional_truncate_map.keys()) + ("extra",),
        prediction_mask_field="target_mask",
        expected_ndim=2 if patch_size is None else 3,
    )
    np.random.seed(0)
    transformed_data_entry = masked_prediction(data_entry.copy())
    target_mask = transformed_data_entry["target_mask"]

    # check target_mask schema
    assert "target_mask" in transformed_data_entry
    assert target_mask.ndim == 2
    assert target_mask.shape[0] == target_dim
    assert target_mask.shape[1] == length
    assert target_mask.dtype == np.dtype(bool)

    # check target_mask values
    mask_ratio = target_mask.sum() / np.prod(target_mask.shape)
    min_mask_ratio = max(1, round(length * min_mask_ratio)) / length
    max_mask_ratio = max(1, round(length * max_mask_ratio)) / length
    assert min_mask_ratio <= mask_ratio <= max_mask_ratio
    num_mask = int(mask_ratio * length)
    assert np.all(target_mask[:, -num_mask:] == np.ones((target_dim, num_mask)))

    # check truncate
    for field, dim in (truncate_map | optional_truncate_map).items():
        if truncate_list:
            assert len(transformed_data_entry[field]) == 2
            for idx, a in enumerate(transformed_data_entry[field]):
                assert (
                    a.shape
                    == (
                        dim,
                        length - num_mask,
                    )
                    + patch_dim
                )
                assert np.allclose(
                    a,
                    data_entry[field][idx][:, :-num_mask],
                )
        else:
            assert (
                transformed_data_entry[field].shape
                == (
                    dim,
                    length - num_mask,
                )
                + patch_dim
            )
            assert np.allclose(
                transformed_data_entry[field],
                data_entry[field][:, :-num_mask],
            )


@pytest.mark.parametrize(
    "field_map",
    [
        dict(target=(1, 10)),
        dict(target=(2, 10)),
        dict(target=(1, 10), another=(2, 9)),
    ],
)
@pytest.mark.parametrize("optional_map", [dict(), dict(optional=(2, 3))])
@pytest.mark.parametrize("patch_size", [None, 2])
def test_extend_mask(
    field_map: dict[str, tuple[int, int]],
    optional_map: dict[str, tuple[int, int]],
    patch_size: Optional[int],
    target_mask_shape: tuple[int, int] = (2, 1),
):
    if patch_size is not None:
        target_mask_shape = target_mask_shape + (patch_size,)
    data_entry = {
        "prediction_mask": np.zeros(target_mask_shape, dtype=bool),
    }
    for field, (dim, length) in (field_map | optional_map).items():
        shape = (dim, length)
        if patch_size is not None:
            shape = shape + (patch_size,)
        data_entry[field] = np.random.randn(*shape)
    extend_mask = ExtendMask(
        fields=tuple(field_map.keys()),
        optional_fields=tuple(optional_map.keys()) + ("extra",),
        mask_field="prediction_mask",
        expected_ndim=2 if patch_size is None else 3,
    )
    transformed_data_entry = extend_mask(data_entry.copy())
    output = transformed_data_entry["prediction_mask"]
    assert len(output) == 1 + len(field_map) + len(optional_map)
    for target_mask, (field, (dim, length)) in zip(
        output[1:], (field_map | optional_map).items()
    ):
        assert target_mask.shape[0] == dim
        assert target_mask.shape[1] == length
        assert target_mask.dtype == np.dtype(bool)
        assert np.all(target_mask == 0)


@pytest.mark.parametrize("length, target_dim", [(5, 1), (10, 2)])
@pytest.mark.parametrize("mask_length", [1, 2])
@pytest.mark.parametrize(
    "truncate_map",
    [
        dict(another=1),
        dict(another1=2, another2=3),
    ],
)
@pytest.mark.parametrize(
    "optional_truncate_map",
    [
        dict(),
        dict(optional=2),
    ],
)
@pytest.mark.parametrize("patch_size", [None, 2])
@pytest.mark.parametrize("truncate_list", [False, True])
def test_eval_masked_prediction(
    length: int,
    target_dim: int,
    mask_length: int,
    truncate_map: dict[str, int],
    optional_truncate_map: dict[str, int],
    patch_size: Optional[int],
    truncate_list: bool,
):
    patch_dim = () if patch_size is None else (patch_size,)
    data_entry = {"target": np.random.randn(*((target_dim, length) + patch_dim))}
    for field, dim in (truncate_map | optional_truncate_map).items():
        if truncate_list:
            data_entry[field] = [
                np.random.randn(*((dim, length) + patch_dim)) for _ in range(2)
            ]
        else:
            data_entry[field] = np.random.randn(*((dim, length) + patch_dim))
    masked_prediction = EvalMaskedPrediction(
        mask_length,
        target_field="target",
        truncate_fields=tuple(truncate_map.keys()),
        optional_truncate_fields=tuple(optional_truncate_map.keys()) + ("extra",),
        prediction_mask_field="target_mask",
        expected_ndim=2 if patch_size is None else 3,
    )
    np.random.seed(0)
    transformed_data_entry = masked_prediction(data_entry.copy())
    target_mask = transformed_data_entry["target_mask"]

    # check target_mask schema
    assert "target_mask" in transformed_data_entry
    assert target_mask.ndim == 2
    assert target_mask.shape[0] == target_dim
    assert target_mask.shape[1] == length
    assert target_mask.dtype == np.dtype(bool)

    # check target_mask values
    assert np.all(target_mask[:, -mask_length:] == np.ones((target_dim, mask_length)))

    # check truncate
    for field, dim in (truncate_map | optional_truncate_map).items():
        if truncate_list:
            assert len(transformed_data_entry[field]) == 2
            for idx, a in enumerate(transformed_data_entry[field]):
                assert (
                    a.shape
                    == (
                        dim,
                        length - mask_length,
                    )
                    + patch_dim
                )
                assert np.allclose(
                    a,
                    data_entry[field][idx][:, :-mask_length],
                )
        else:
            assert (
                transformed_data_entry[field].shape
                == (
                    dim,
                    length - mask_length,
                )
                + patch_dim
            )
            assert np.allclose(
                transformed_data_entry[field],
                data_entry[field][:, :-mask_length],
            )
