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
import string
from functools import partial
from pathlib import Path
from typing import Any, Callable, Generator, Optional, Type

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset

from uni2ts.common.typing import FlattenedData
from uni2ts.data.builder import DatasetBuilder
from uni2ts.data.builder.lotsa_v1 import LOTSADatasetBuilder
from uni2ts.data.dataset import TimeSeriesDataset


@pytest.fixture(scope="session")
def create_data_entry() -> Callable[..., dict[str, FlattenedData]]:
    def _create_data_entry(
        length: int,
        freq: str,
        target_dim: int = 1,
        past_feat_dynamic_real_dim: Optional[int] = None,
        item_id: str = "item_1",
    ) -> dict[str, FlattenedData]:
        assert length > 0
        assert target_dim > 0
        assert past_feat_dynamic_real_dim is None or past_feat_dynamic_real_dim > 0

        data_entry = dict(
            target=(
                [np.random.randn(length)]
                if target_dim == 1
                else list(np.random.randn(target_dim, length))
            ),
            item_id=item_id,
            freq=freq,
            start=np.asarray(np.datetime64("2020-01-01T00:00")),
        )

        if past_feat_dynamic_real_dim is not None:
            data_entry["past_feat_dynamic_real"] = (
                [np.random.randn(length)]
                if target_dim == 1
                else list(np.random.randn(past_feat_dynamic_real_dim, length))
            )

        return data_entry

    return _create_data_entry


@pytest.fixture(scope="session")
def create_example() -> Callable[..., dict[str, ...]]:
    def _create_example(
        length: int,
        freq: str,
        target_dim: int = 1,
        past_feat_dynamic_real_dim: Optional[int] = None,
        item_id: str = "item_1",
    ) -> dict[str, FlattenedData]:
        assert length > 0
        assert target_dim > 0
        assert past_feat_dynamic_real_dim is None or past_feat_dynamic_real_dim > 0

        data_entry = dict(
            target=(
                np.random.randn(length)
                if target_dim == 1
                else np.random.randn(target_dim, length)
            ),
            item_id=item_id,
            freq=freq,
            start=pd.Timestamp(np.datetime64("2020-01-01T00:00")),
        )

        if past_feat_dynamic_real_dim is not None:
            data_entry["past_feat_dynamic_real"] = (
                np.random.randn(length)
                if past_feat_dynamic_real_dim == 1
                else np.random.randn(past_feat_dynamic_real_dim, length)
            )

        return data_entry

    return _create_example


@pytest.fixture(scope="session")
def parquet_files(
    tmp_path_factory, create_data_entry, request
) -> Generator[tuple[list[Path], int, int, Optional[int], tuple[int, int]], None, None]:
    path = tmp_path_factory.mktemp("parquet_files")

    num_examples: int = request.param[0]
    target_dim: int = request.param[1]
    past_feat_dynamic_real_dim: Optional[int] = request.param[2]
    length: tuple[int, int] = request.param[3]

    rng = np.random.default_rng(0)
    ts_lengths = [rng.integers(*length, endpoint=True) for _ in range(num_examples)]
    files = []

    for idx in range(3):

        def gen() -> Generator[dict[str, Any], None, None]:
            data_entry = create_data_entry(
                length=ts_lengths[idx],
                freq="H",
                target_dim=target_dim,
                past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
                item_id=f"item_{idx}",
            )
            yield data_entry

        dataset = Dataset.from_generator(gen)
        dataset.to_parquet(path / f"item_{idx}.parquet")
        dataset.cleanup_cache_files()
        files.append(path / f"item_{idx}.parquet")

    yield files, num_examples, target_dim, past_feat_dynamic_real_dim, length
    shutil.rmtree(path)


@pytest.fixture(scope="session")
def hf_dataset_path(
    tmp_path_factory, create_example, request
) -> Generator[tuple[Path, int, int, Optional[int], tuple[int, int]], None, None]:
    path = tmp_path_factory.mktemp("arrow_files")

    num_examples: int = request.param[0]
    target_dim: int = request.param[1]
    past_feat_dynamic_real_dim: Optional[int] = request.param[2]
    length: tuple[int, int] = request.param[3]

    rng = np.random.default_rng(0)

    def gen_func() -> Generator[dict[str, Any], None, None]:
        for idx in range(num_examples):
            data_entry = create_example(
                length=rng.integers(*length, endpoint=True),
                freq="H",
                target_dim=target_dim,
                past_feat_dynamic_real_dim=past_feat_dynamic_real_dim,
                item_id=f"item_{idx}",
            )
            yield data_entry

    hf_dataset = Dataset.from_generator(gen_func)
    hf_dataset.save_to_disk(path / "hf_dataset")

    yield path / "hf_dataset", num_examples, target_dim, past_feat_dynamic_real_dim, length
    shutil.rmtree(str(path))


class AirPassengersDatasetBuilder(LOTSADatasetBuilder):
    dataset_list: list[str] = ["airpassengers"]
    dataset_type_map: dict[str, Type[TimeSeriesDataset]] = {
        "airpassengers": TimeSeriesDataset,
    }
    dataset_load_func_map: dict[str, Callable[..., Dataset]] = {
        "airpassengers": partial(TimeSeriesDataset),
    }

    def build_dataset(self):
        from gluonts.dataset.repository.datasets import get_dataset

        air_passengers_dataset = get_dataset("airpassengers")

        def example_gen_func():
            for entry in air_passengers_dataset.train:
                entry["freq"] = entry["start"].freqstr
                entry["start"]: pd.Period = entry["start"].to_timestamp("s")
                yield entry

        hf_dataset = Dataset.from_generator(example_gen_func)
        hf_dataset.save_to_disk(self.storage_path / "airpassengers")


@pytest.fixture(scope="session")
def airpassengers_dataset_builder(
    tmp_path_factory, request
) -> Generator[tuple[DatasetBuilder, Optional[float], Optional[int]], None, None]:
    path = tmp_path_factory.mktemp("airpassengers")
    weight: Optional[float] = request.param[0]
    repeat: Optional[int] = request.param[1]
    weight_map = None if weight is None else dict(airpassengers=weight)
    dataset_builder = AirPassengersDatasetBuilder(
        datasets=["airpassengers"] * (repeat or 1),
        weight_map=weight_map,
        storage_path=path,
    )
    dataset_builder.build_dataset()
    yield dataset_builder, weight, repeat
    shutil.rmtree(str(path))


@pytest.fixture(scope="session")
def get_wide_df(
    tmp_path_factory, request
) -> Generator[tuple[Path, int, int], None, None]:
    path = tmp_path_factory.mktemp("wide_df")
    filepath = path / "wide_df.csv"
    num_columns: int = request.param[0]
    num_rows: int = request.param[1]

    df = pd.DataFrame(
        index=np.datetime64("2020-01-01") + np.arange(num_rows),
        data={
            col: np.random.randn(num_rows)
            for col, _ in zip(string.ascii_uppercase, range(num_columns))
        },
    )
    df.to_csv(filepath)
    yield filepath, num_columns, num_rows
    shutil.rmtree(str(path))


@pytest.fixture(scope="session")
def get_long_df(
    tmp_path_factory, request
) -> Generator[tuple[Path, int, int], None, None]:
    path = tmp_path_factory.mktemp("long_df")
    filepath = path / "long_df.csv"
    num_columns: int = request.param[0]
    num_rows: int = request.param[1]

    df = pd.DataFrame(
        index=np.concatenate(
            [np.datetime64("2020-01-01") + np.arange(num_rows)] * num_columns
        ),
        data={
            "item_id": np.concatenate(
                [
                    np.asarray([col] * num_rows)
                    for col, _ in zip(string.ascii_uppercase, range(num_columns))
                ]
            ),
            "target": np.random.randn(num_columns * num_rows),
        },
    )
    df.to_csv(filepath)
    yield filepath, num_columns, num_rows
    shutil.rmtree(str(path))
