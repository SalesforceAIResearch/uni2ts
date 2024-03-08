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

from uni2ts.transform.resample import SampleDimension


@pytest.mark.parametrize("target_dim", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("past_feat_dynamic_real_dim", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("seed", [i for i in range(10)])
def test_sample_dimension_shape(
    create_data_entry,
    target_dim: int,
    past_feat_dynamic_real_dim: Optional[int],
    seed: int,
):
    data_entry = create_data_entry(
        10,
        "H",
        target_dim=target_dim,
        past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
    )
    np.random.seed(0)
    sample_dimension = SampleDimension(
        max_dim=target_dim + (past_feat_dynamic_real_dim or 0),
        fields=("target",),
        optional_fields=("past_feat_dynamic_real",),
    )
    transformed_data_entry = sample_dimension(data_entry.copy())

    np.random.seed(0)
    for f in ["target", "past_feat_dynamic_real"]:
        assert data_entry[f][0].shape[0] == transformed_data_entry[f][0].shape[0], (
            f"Time length of {f} should not change from "
            f"{data_entry[f][0].shape[0]} to {transformed_data_entry[f][0].shape[0]}"
        )
        out = np.random.permutation(len(data_entry[f]))
        n = sample_dimension.sampler(out.shape[0])

        assert len(data_entry[f]) >= len(transformed_data_entry[f]), (
            f"Dimension of transformed {f} should be less than the original. "
            f"Original: {len(data_entry[f])}, Transformed: {len(transformed_data_entry[f])}"
        )
        assert (
            len(transformed_data_entry[f]) == n
        ), f"Dimension of transformed {f} should be {n}. "
