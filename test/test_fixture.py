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

import pytest

from uni2ts.common.typing import UnivarTimeSeries


@pytest.mark.parametrize(
    "length, freq, target_dim, past_feat_dynamic_real_dim",
    [(10, "H", 1, None), (10, "H", 2, None), (10, "H", 2, 1), (10, "H", 2, 2)],
)
def test_create_data_entry(
    create_data_entry,
    length: int,
    freq: str,
    target_dim: int,
    past_feat_dynamic_real_dim: Optional[int],
):
    data_entry = create_data_entry(length, freq, target_dim, past_feat_dynamic_real_dim)

    assert isinstance(data_entry["target"], list)
    assert len(data_entry["target"]) == target_dim
    assert all(isinstance(ts, UnivarTimeSeries) for ts in data_entry["target"])
    assert all(len(ts) == length for ts in data_entry["target"])

    if past_feat_dynamic_real_dim is None:
        assert "past_feat_dynamic_real" not in data_entry
    else:
        assert isinstance(data_entry["past_feat_dynamic_real"], list)
        assert len(data_entry["past_feat_dynamic_real"]) == past_feat_dynamic_real_dim
        assert all(
            isinstance(ts, UnivarTimeSeries)
            for ts in data_entry["past_feat_dynamic_real"]
        )
        assert all(len(ts) == length for ts in data_entry["past_feat_dynamic_real"])


@pytest.mark.parametrize(
    "length, freq, target_dim, past_feat_dynamic_real_dim",
    [(10, "H", 1, None), (10, "H", 2, None), (10, "H", 2, 1), (10, "H", 2, 2)],
)
def test_create_example(
    create_example,
    length: int,
    freq: str,
    target_dim: int,
    past_feat_dynamic_real_dim: Optional[int],
):
    data_entry = create_example(length, freq, target_dim, past_feat_dynamic_real_dim)

    target_shape = (length,) if target_dim == 1 else (target_dim, length)
    assert data_entry["target"].shape == target_shape

    if past_feat_dynamic_real_dim is None:
        assert "past_feat_dynamic_real" not in data_entry
    else:
        feat_shape = (
            (length,)
            if past_feat_dynamic_real_dim == 1
            else (past_feat_dynamic_real_dim, length)
        )
        assert data_entry["past_feat_dynamic_real"].shape == feat_shape
