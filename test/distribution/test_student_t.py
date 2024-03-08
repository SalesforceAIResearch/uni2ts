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

import pytest
import torch

from uni2ts.distribution.student_t import StudentTOutput


@pytest.mark.parametrize("batch_shape", [tuple(), (1,), (2, 3)])
@pytest.mark.parametrize("in_features", [32, 64])
@pytest.mark.parametrize("out_features_ls", [(10,), (10, 20, 30)])
@pytest.mark.parametrize("sample_shape", [tuple(), (1,), (2, 3)])
def test_student_t_output_shape(
    batch_shape: tuple[int, ...],
    in_features: int,
    out_features_ls: tuple[int, ...],
    sample_shape: tuple[int, ...],
):
    student_t_output = StudentTOutput()
    param_proj = student_t_output.get_param_proj(in_features, out_features_ls)

    x = torch.randn(batch_shape + (in_features,))
    out_feat_size = torch.randint(1, max(out_features_ls) + 1, batch_shape)
    distr_param = param_proj(x, out_feat_size)

    distr = student_t_output.distribution(distr_param)
    sample = distr.sample(torch.Size(sample_shape))
    assert sample.shape == sample_shape + batch_shape + (max(out_features_ls),)
