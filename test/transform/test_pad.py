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

from uni2ts.transform.pad import EvalPad, Pad, PadFreq


@pytest.mark.parametrize(
    "min_length, target_length",
    [
        (10, 10),
        (20, 10),
        (10, 20),
    ],
)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_pad(
    min_length: int,
    target_length: int,
    dim: int,
):
    data_entry = {
        "target": np.random.randn(
            *((dim, target_length) if dim > 1 else (target_length,))
        ),
    }
    transformed_data_entry = Pad(
        min_length=min_length,
        fields=("target",),
    )(data_entry.copy())

    assert transformed_data_entry["target"].shape[-1] == max(min_length, target_length)
    if dim > 1:
        assert transformed_data_entry["target"].shape[0] == dim
    else:
        assert len(transformed_data_entry["target"].shape) == 1


@pytest.mark.parametrize(
    "min_length, target_length",
    [
        (10, 10),
        (20, 10),
        (10, 20),
    ],
)
@pytest.mark.parametrize("dim", [1, 2, 3])
def test_pad_freq(
    min_length: int,
    target_length: int,
    dim: int,
):
    data_entry = {
        "target": np.random.randn(
            *((dim, target_length) if dim > 1 else (target_length,))
        ),
        "freq": "h",
    }

    transformed_data_entry = PadFreq(
        freq_min_length_map={"h": min_length},
        fields=("target",),
        freq_field="freq",
    )(data_entry.copy())

    assert transformed_data_entry["target"].shape[-1] == max(min_length, target_length)
    if dim > 1:
        assert transformed_data_entry["target"].shape[0] == dim
    else:
        assert len(transformed_data_entry["target"].shape) == 1


@pytest.mark.parametrize("prediction_length", [100, 1000])
@pytest.mark.parametrize("context_length", [100, 1000])
@pytest.mark.parametrize("patch_size", [8, 16])
@pytest.mark.parametrize("dim", [1, 3])
def test_patch_eval_pad(
    prediction_length: int,
    context_length: int,
    patch_size: int,
    dim: int,
):
    data_entry = {
        "target": np.random.randn(
            *(
                (dim, context_length + prediction_length)
                if dim > 1
                else (context_length + prediction_length,)
            )
        ),
    }
    eval_pad = EvalPad(
        context_pad=-context_length % patch_size,
        prediction_pad=-prediction_length % patch_size,
        fields=("target",),
    )
    transformed_data_entry = eval_pad(data_entry.copy())

    length = transformed_data_entry["target"].shape[-1]
    front_pad = -context_length % patch_size
    assert length >= context_length + prediction_length
    assert length % patch_size == 0
    assert np.all(
        transformed_data_entry["target"][..., front_pad : front_pad + context_length]
        == data_entry["target"][..., :context_length]
    )
    assert np.all(
        transformed_data_entry["target"][
            ...,
            front_pad + context_length : front_pad + context_length + prediction_length,
        ]
        == data_entry["target"][..., context_length:]
    )
    if dim > 1:
        assert transformed_data_entry["target"].shape[0] == dim
    else:
        assert len(transformed_data_entry["target"].shape) == 1
