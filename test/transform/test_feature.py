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
from typing import Optional

import numpy as np
import pytest

from uni2ts.transform.feature import AddObservedMask, AddTimeIndex, AddVariateIndex


@pytest.mark.parametrize(
    "field_map",
    [
        dict(target=(1, 10)),
        dict(target=(2, 10)),
        dict(target=(1, 10), another=(2, 9)),
    ],
)
@pytest.mark.parametrize("optional_map", [dict(), dict(optional=(2, 3))])
@pytest.mark.parametrize("max_dim", [1, 5])
@pytest.mark.parametrize("patch_size", [None, 2])
@pytest.mark.parametrize("randomize", [False, True])
@pytest.mark.parametrize("collection_type", [list, dict])
def test_add_variate_index(
    field_map: dict[str, tuple[int, int]],
    optional_map: dict[str, tuple[int, int]],
    max_dim: int,
    patch_size: Optional[int],
    randomize: bool,
    collection_type: type,
):
    if max_dim < sum([dim for dim, _ in (field_map | optional_map).values()]):
        expectation = pytest.raises(
            ValueError, match=r"Variate \(\d+\) exceeds maximum variate (\d+)."
        )
    else:
        expectation = does_not_raise()
    with expectation:
        data_entry = {}
        for field, (dim, length) in (field_map | optional_map).items():
            shape = (dim, length)
            if patch_size is not None:
                shape = shape + (patch_size,)
            data_entry[field] = np.random.randn(*shape)

        add_variate_index = AddVariateIndex(
            fields=tuple(field_map.keys()),
            max_dim=max_dim,
            optional_fields=tuple(optional_map.keys()) + ("extra",),
            variate_id_field="variate_id",
            expected_ndim=2 if patch_size is None else 3,
            randomize=randomize,
            collection_type=collection_type,
        )
        transformed_data_entry = add_variate_index(data_entry.copy())
        output = transformed_data_entry["variate_id"]
        variate_ids = set()
        if collection_type == list:
            assert len(output) == len(field_map) + len(optional_map)
            for dim_id, (field, (dim, length)) in zip(
                output, (field_map | optional_map).items()
            ):
                assert dim_id.shape[0] == dim
                assert dim_id.shape[1] == length
                for d in range(dim):
                    assert np.allclose(dim_id[d, :-1], dim_id[d, 1:])
                    assert dim_id[d, 0].item() not in variate_ids
                    assert dim_id[d, 0].item() in range(max_dim)
                    variate_ids.add(dim_id[d, 0].item())
        else:
            assert len(output) == len(field_map) + len(optional_map)
            for field, (dim, length) in (field_map | optional_map).items():
                assert output[field].shape[0] == dim
                assert output[field].shape[1] == length
                for d in range(dim):
                    assert np.allclose(output[field][d, :-1], output[field][d, 1:])
                    assert output[field][d, 0].item() not in variate_ids
                    assert output[field][d, 0].item() in range(max_dim)
                    variate_ids.add(output[field][d, 0].item())


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
@pytest.mark.parametrize("collection_type", [list, dict])
def test_add_time_index(
    field_map: dict[str, tuple[int, int]],
    optional_map: dict[str, tuple[int, int]],
    patch_size: Optional[int],
    collection_type: type,
):
    data_entry = {}
    for field, (dim, length) in (field_map | optional_map).items():
        shape = (dim, length)
        if patch_size is not None:
            shape = shape + (patch_size,)
        data_entry[field] = np.random.randn(*shape)

    add_time_index = AddTimeIndex(
        fields=tuple(field_map.keys()),
        optional_fields=tuple(optional_map.keys()) + ("extra",),
        time_id_field="time_id",
        expected_ndim=2 if patch_size is None else 3,
        collection_type=collection_type,
    )
    transformed_data_entry = add_time_index(data_entry.copy())
    output = transformed_data_entry["time_id"]
    if collection_type == list:
        assert len(output) == len(field_map) + len(optional_map)
        for seq_id, (field, (dim, length)) in zip(
            output, (field_map | optional_map).items()
        ):
            assert seq_id.shape[0] == dim
            assert seq_id.shape[1] == length
            for d in range(dim):
                assert np.all(seq_id[d] == np.arange(length))
    else:
        assert len(output) == len(field_map) + len(optional_map)
        for field, (dim, length) in (field_map | optional_map).items():
            assert output[field].shape[0] == dim
            assert output[field].shape[1] == length
            for d in range(dim):
                assert np.all(output[field][d] == np.arange(length))


@pytest.mark.parametrize(
    "field_map",
    [
        dict(target=(1, 10)),
        dict(target=(2, 10)),
        dict(target=(1, 10), another=(2, 9)),
    ],
)
@pytest.mark.parametrize("optional_map", [dict(), dict(optional=(2, 3))])
@pytest.mark.parametrize("collection_type", [list, dict])
def test_add_observed_mask(
    field_map: dict[str, tuple[int, int]],
    optional_map: dict[str, tuple[int, int]],
    collection_type: type,
):
    data_entry = {}
    nans = {}
    for field, (dim, length) in (field_map | optional_map).items():
        shape = (dim, length)
        data_entry[field] = np.random.randn(*shape)
        nans[field] = np.random.rand(*shape) < 0.5
        data_entry[field][nans[field]] = np.nan

    add_observed_mask = AddObservedMask(
        fields=tuple(field_map.keys()),
        optional_fields=tuple(optional_map.keys()) + ("extra",),
        observed_mask_field="observed_mask",
        collection_type=collection_type,
    )
    transformed_data_entry = add_observed_mask(data_entry.copy())
    output = transformed_data_entry["observed_mask"]
    if collection_type == list:
        assert len(output) == len(field_map) + len(optional_map)
        for mask, (field, (dim, length)) in zip(
            output, (field_map | optional_map).items()
        ):
            assert mask.shape[0] == dim
            assert mask.shape[1] == length
            assert np.all(mask == ~nans[field])
    else:
        assert len(output) == len(field_map) + len(optional_map)
        for field, (dim, length) in (field_map | optional_map).items():
            assert output[field].shape[0] == dim
            assert output[field].shape[1] == length
            assert np.all(output[field] == ~nans[field])
