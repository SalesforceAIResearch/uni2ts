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
from torch.utils.data import ConcatDataset

from uni2ts.data.dataset import TimeSeriesDataset


@pytest.mark.parametrize(
    "airpassengers_dataset_builder",
    [(None, None), (1.0, 1), (1.5, 2), (2.0, 2)],
    indirect=True,
)
def test_dataset_builder(
    airpassengers_dataset_builder,
):
    dataset_builder, weight, repeat = airpassengers_dataset_builder
    dataset = dataset_builder.load_dataset({})
    if (repeat or 1) > 1:
        assert isinstance(dataset, ConcatDataset)
        for sub_dataset in dataset.datasets:
            assert isinstance(sub_dataset, TimeSeriesDataset)
    else:
        assert isinstance(dataset, TimeSeriesDataset)
    assert len(dataset) == int(np.ceil(1 * (weight or 1.0))) * (repeat or 1)
