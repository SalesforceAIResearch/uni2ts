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

import numpy as np
import pytest

from uni2ts.transform import DummyValueImputation, ImputeTimeSeries, LastValueImputation


@pytest.mark.parametrize(
    "x, y, value",
    [
        (np.asarray([1.0, 2.0, np.nan, 4.0]), np.asarray([1.0, 2.0, 0.0, 4.0]), 0.0),
        (
            np.asarray(
                [
                    [np.nan, 2, 3, np.nan],
                    [1, 2, np.nan, 4],
                ]
            ),
            np.asarray(
                [
                    [-99, 2, 3, -99],
                    [1, 2, -99, 4],
                ]
            ),
            -99,
        ),
    ],
)
@pytest.mark.parametrize(
    "fields",
    [("target",), ("past_feat_dynamic_real",), ("target", "past_feat_dynamic_real")],
)
def test_dummy_value_imputation(
    x: np.ndarray, y: np.ndarray, value: int | float, fields: tuple[str, ...]
):
    # should be (time, dim)
    x = x.T
    y = y.T

    transform = ImputeTimeSeries(
        imputation_method=DummyValueImputation(value=value),
        fields=fields,
    )
    data_entry = {field: x for field in fields}
    transformed_data_entry = transform(data_entry)
    for field in fields:
        assert np.all(transformed_data_entry[field] == y)


@pytest.mark.parametrize(
    "x, y, value",
    [
        (np.asarray([1.0, 2.0, np.nan, 4.0]), np.asarray([1.0, 2.0, 2.0, 4.0]), 0),
        (
            np.asarray(
                [
                    [np.nan, 2, 3, np.nan],
                    [1, 2, np.nan, 4],
                ]
            ),
            np.asarray(
                [
                    [-99, 2, 3, 3],
                    [1, 2, 2, 4],
                ]
            ),
            -99,
        ),
    ],
)
@pytest.mark.parametrize(
    "fields",
    [("target",), ("past_feat_dynamic_real",), ("target", "past_feat_dynamic_real")],
)
def test_last_value_imputation(
    x: np.ndarray, y: np.ndarray, value: int | float, fields: tuple[str, ...]
):
    transform = ImputeTimeSeries(
        imputation_method=LastValueImputation(value=value),
        fields=fields,
    )
    data_entry = {field: x for field in fields}
    transformed_data_entry = transform(data_entry)
    for field in fields:
        assert np.all(transformed_data_entry[field] == y)
