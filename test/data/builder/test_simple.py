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

import shutil
from typing import Optional

import datasets
import pandas as pd
import pytest

from uni2ts.data.builder.simple import SimpleDatasetBuilder


@pytest.mark.parametrize(
    "get_wide_df",
    [(1, 10), (3, 10), (3, 100)],
    indirect=True,
)
@pytest.mark.parametrize("multivariate", [True, False])
@pytest.mark.parametrize(
    "offset, date_offset",
    [
        (7, None),
        (-3, None),
        (None, pd.Timestamp("2020-01-03")),
        (None, pd.Timestamp("2020-01-07")),
    ],
)
def test_simple_dataset_builder_wide(
    tmp_path_factory,
    get_wide_df,
    multivariate: bool,
    offset: Optional[int],
    date_offset: Optional[pd.Timestamp],
):
    storage_path = tmp_path_factory.mktemp("storage")
    wide_df_path, num_columns, num_rows = get_wide_df
    builder = SimpleDatasetBuilder(
        "wide_dataset",
        storage_path=storage_path,
    )
    builder.build_dataset(
        file=wide_df_path,
        dataset_type="wide" + ("_multivariate" if multivariate else ""),
        offset=offset,
        date_offset=date_offset,
    )

    assert (storage_path / "wide_dataset").is_dir()

    hf_dataset = datasets.load_from_disk(
        str(storage_path / "wide_dataset")
    ).with_format("numpy")

    if offset is not None:
        if offset > 0:
            time = offset
        else:
            time = num_rows + offset
    else:
        time = (date_offset - pd.Timestamp("2020-01-01")).days + 1

    if multivariate:
        assert len(hf_dataset) == 1
        assert hf_dataset[0]["target"].shape == (num_columns, time)
    else:
        assert len(hf_dataset) == num_columns
        assert hf_dataset[0]["target"].shape == (time,)

    shutil.rmtree(str(storage_path))


@pytest.mark.parametrize(
    "get_long_df",
    [(1, 10), (3, 10), (3, 100)],
    indirect=True,
)
@pytest.mark.parametrize(
    "offset, date_offset",
    [
        (7, None),
        (-3, None),
        (None, pd.Timestamp("2020-01-03")),
        (None, pd.Timestamp("2020-01-07")),
    ],
)
def test_simple_dataset_builder_long(
    tmp_path_factory,
    get_long_df,
    offset: Optional[int],
    date_offset: Optional[pd.Timestamp],
):
    storage_path = tmp_path_factory.mktemp("storage")
    wide_df_path, num_columns, num_rows = get_long_df
    builder = SimpleDatasetBuilder(
        "long_dataset",
        storage_path=storage_path,
    )
    builder.build_dataset(
        file=wide_df_path,
        dataset_type="long",
        offset=offset,
        date_offset=date_offset,
    )

    assert (storage_path / "long_dataset").is_dir()

    hf_dataset = datasets.load_from_disk(
        str(storage_path / "long_dataset")
    ).with_format("numpy")

    if offset is not None:
        if offset > 0:
            time = offset
        else:
            time = num_rows + offset
    else:
        time = (date_offset - pd.Timestamp("2020-01-01")).days + 1

    assert len(hf_dataset) == num_columns
    assert hf_dataset[0]["target"].shape == (time,)

    shutil.rmtree(str(storage_path))
