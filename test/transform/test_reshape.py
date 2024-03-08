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

from uni2ts.transform import GetPatchSize, PatchCrop, Patchify, SequencifyField
from uni2ts.transform.reshape import (
    FlatPackCollection,
    FlatPackFields,
    PackCollection,
    PackFields,
)


@pytest.mark.parametrize("length", [1, 10, 20])
@pytest.mark.parametrize("output_field", ["target", "output"])
@pytest.mark.parametrize(
    "field_dim_mapping, optional_field_dim_mapping",
    [
        (dict(target=1), dict(past_feat_dynamic_real=1)),
        (dict(target=1), dict(past_feat_dynamic_real=3)),
        (dict(target=2), dict(past_feat_dynamic_real=3)),
        (dict(output=3), dict()),
    ],
)
@pytest.mark.parametrize("patch_size", [None, 2])
def test_pack_fields(
    length: int,
    output_field: str,
    field_dim_mapping: dict[str, int],
    optional_field_dim_mapping: dict[str, int],
    patch_size: Optional[int],
):
    data_entry = {}
    for field, dim in (field_dim_mapping | optional_field_dim_mapping).items():
        shape = (length,)
        if dim > 1:
            shape = (dim,) + shape
        if patch_size is not None:
            shape = shape + (patch_size,)
        data_entry[field] = np.random.randn(*shape)

    pack_fields = PackFields(
        output_field=output_field,
        fields=tuple(field_dim_mapping.keys()),
        optional_fields=tuple(optional_field_dim_mapping.keys()) + ("extra",),
        feat=patch_size is not None,
    )
    transformed_data_entry = pack_fields(data_entry.copy())
    output = transformed_data_entry[output_field]
    assert output.shape[0] == sum(
        (field_dim_mapping | optional_field_dim_mapping).values()
    )
    assert output.shape[1] == length
    if patch_size is not None:
        assert output.shape[-1] == patch_size
    else:
        assert output.ndim == 2
    cum_dim = 0
    for field, dim in (field_dim_mapping | optional_field_dim_mapping).items():
        for d in range(dim):
            if dim == 1:
                data = data_entry[field]
            else:
                data = data_entry[field][d]
            assert np.allclose(
                data,
                output[cum_dim],
            )
            cum_dim += 1


@pytest.mark.parametrize("length", [1, 10, 20])
@pytest.mark.parametrize("output_field", ["target", "output"])
@pytest.mark.parametrize(
    "field_dim_mapping, optional_field_dim_mapping",
    [
        (dict(target=1), dict(past_feat_dynamic_real=1)),
        (dict(target=1), dict(past_feat_dynamic_real=3)),
        (dict(target=2), dict(past_feat_dynamic_real=3)),
        (dict(output=3), dict()),
    ],
)
@pytest.mark.parametrize("patch_size", [None, 2])
def test_flat_pack_fields(
    length: int,
    output_field: str,
    field_dim_mapping: dict[str, int],
    optional_field_dim_mapping: dict[str, int],
    patch_size: Optional[int],
):
    data_entry = {}
    for field, dim in (field_dim_mapping | optional_field_dim_mapping).items():
        shape = (length,)
        if dim > 1:
            shape = (dim,) + shape
        if patch_size is not None:
            shape = shape + (patch_size,)
        data_entry[field] = np.random.randn(*shape)

    pack_fields = FlatPackFields(
        output_field=output_field,
        fields=tuple(field_dim_mapping.keys()),
        optional_fields=tuple(optional_field_dim_mapping.keys()) + ("extra",),
        feat=patch_size is not None,
    )
    transformed_data_entry = pack_fields(data_entry.copy())
    output = transformed_data_entry[output_field]
    assert output.shape[0] == length * sum(
        (field_dim_mapping | optional_field_dim_mapping).values()
    )
    if patch_size is not None:
        assert output.ndim == 2
        assert output.shape[-1] == patch_size
    else:
        assert output.ndim == 1
    cum_dim = 0
    for field, dim in (field_dim_mapping | optional_field_dim_mapping).items():
        for d in range(dim):
            if dim == 1:
                data = data_entry[field]
            else:
                data = data_entry[field][d]
            assert np.allclose(
                data,
                output[cum_dim * length : (cum_dim + 1) * length],
            )
            cum_dim += 1


@pytest.mark.parametrize(
    "length, dims",
    [
        (1, [1]),
        (1, [2]),
        (10, [2]),
        (10, [1, 2]),
    ],
)
@pytest.mark.parametrize("patch_size", [None, 2])
def test_pack_collection(
    length: int,
    dims: list[int],
    patch_size: Optional[int],
):
    data_entry = {"target": []}
    for dim in dims:
        shape = (length,)
        if dim > 1:
            shape = (dim,) + shape
        if patch_size is not None:
            shape = shape + (patch_size,)
        data_entry["target"].append(np.random.randn(*shape))

    pack_fields = PackCollection(
        field="target",
        feat=patch_size is not None,
    )
    transformed_data_entry = pack_fields(data_entry.copy())
    output = transformed_data_entry["target"]
    assert output.shape[0] == sum(dims)
    assert output.shape[1] == length
    if patch_size is not None:
        assert output.ndim == 3
        assert output.shape[-1] == patch_size
    else:
        assert output.ndim == 2
    cum_dim = 0
    for i, dim in enumerate(dims):
        for d in range(dim):
            if dim == 1:
                data = data_entry["target"][i]
            else:
                data = data_entry["target"][i][d]
            assert np.allclose(
                data,
                output[cum_dim],
            )
            cum_dim += 1


@pytest.mark.parametrize(
    "lengths, dims",
    [
        ([1], [1]),
        ([1], [2]),
        ([10], [2]),
        ([1, 10], [1, 1]),
        ([1, 10], [1, 2]),
    ],
)
@pytest.mark.parametrize("patch_size", [None, 2])
def test_flat_pack_collection(
    lengths: list[int],
    dims: list[int],
    patch_size: Optional[int],
):
    assert len(lengths) == len(dims)
    data_entry = {"target": []}
    for length, dim in zip(lengths, dims):
        shape = (length,)
        if dim > 1:
            shape = (dim,) + shape
        if patch_size is not None:
            shape = shape + (patch_size,)
        data_entry["target"].append(np.random.randn(*shape))

    pack_fields = FlatPackCollection(
        field="target",
        feat=patch_size is not None,
    )
    transformed_data_entry = pack_fields(data_entry.copy())
    output = transformed_data_entry["target"]
    assert output.shape[0] == sum([length * dim for length, dim in zip(lengths, dims)])
    if patch_size is not None:
        assert output.ndim == 2
        assert output.shape[-1] == patch_size
    else:
        assert output.ndim == 1
    cum_length = 0
    for i, (length, dim) in enumerate(zip(lengths, dims)):
        for d in range(dim):
            if dim == 1:
                data = data_entry["target"][i]
            else:
                data = data_entry["target"][i][d]
            assert np.allclose(
                data,
                output[cum_length : cum_length + length],
            )
            cum_length += length


@pytest.mark.parametrize("length", [1000, 2000, 3000])
@pytest.mark.parametrize("target_dim", [1, 2])
def test_patch_size_sequence(
    create_data_entry,
    length: int,
    target_dim: int,
):
    max_patch_size = 128
    np.random.seed(0)
    fields = ("target",)

    data_entry = create_data_entry(
        length,
        "H",
        target_dim=target_dim,
    )
    data_entry = (
        GetPatchSize(
            min_time_patches=2,
        )
        + PatchCrop(2, 512, fields=fields, will_flatten=False)
        + PackFields("target", fields=("target",))
        + Patchify(max_patch_size, fields=fields)
    )(data_entry)
    transformed_data_entry = SequencifyField(
        field="patch_size", target_field="target", target_axis=1
    )(data_entry.copy())

    patch_size = data_entry["patch_size"]
    transformed_patch_size = transformed_data_entry["patch_size"]

    assert isinstance(transformed_patch_size, np.ndarray)
    assert transformed_patch_size.dtype == int
    assert transformed_patch_size.ndim == 1
    assert transformed_patch_size.shape == (data_entry["target"].shape[1],)
    assert (transformed_patch_size == patch_size).all()
