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

from collections import defaultdict
from functools import partial
from typing import Any, Generator

import datasets
import numpy as np
from datasets import Features, Sequence, Value

try:
    from subseasonal_data import data_loaders
except ImportError as e:

    def data_loaders(*args, **kwargs):
        raise e


from uni2ts.common.env import env
from uni2ts.common.typing import GenFunc
from uni2ts.data.dataset import MultiSampleTimeSeriesDataset

from ._base import LOTSADatasetBuilder


def _get_subseasonal_precip_gen_func() -> tuple[GenFunc, Features]:
    precip = data_loaders.get_ground_truth("us_precip")
    lat_lon = precip[["lat", "lon"]].value_counts().index

    def gen_func() -> Generator[dict[str, Any], None, None]:
        for lat, lon in lat_lon:
            lat_lon_precip = (
                precip.query(f"lat == {lat} and lon == {lon}")
                .set_index("start_date")
                .sort_index()
            )
            yield dict(
                item_id=f"{lat}_{lon}",
                start=lat_lon_precip.index[0],
                target=lat_lon_precip["precip"][:11323].to_numpy(),
                freq="D",
            )

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            freq=Value("string"),
            target=Sequence(Value("float32")),
        )
    )

    return gen_func, features


def _get_subseasonal_gen_func() -> tuple[GenFunc, Features]:
    precip = data_loaders.get_ground_truth("us_precip")
    tmp2m = data_loaders.get_ground_truth("us_tmp2m")
    tmin = data_loaders.get_ground_truth("us_tmin")
    tmax = data_loaders.get_ground_truth("us_tmax")

    lat_lon = precip[["lat", "lon"]].value_counts().index

    def gen_func() -> Generator[dict[str, Any], None, None]:
        for lat, lon in lat_lon:
            lat_lon_precip = (
                precip.query(f"lat == {lat} and lon == {lon}")
                .set_index("start_date")
                .sort_index()
            )
            lat_lon_tmp2m = (
                tmp2m.query(f"lat == {lat} and lon == {lon}")
                .set_index("start_date")
                .sort_index()
            )
            lat_lon_tmin = (
                tmin.query(f"lat == {lat} and lon == {lon}")
                .set_index("start_date")
                .sort_index()
            )
            lat_lon_tmax = (
                tmax.query(f"lat == {lat} and lon == {lon}")
                .set_index("start_date")
                .sort_index()
            )

            yield dict(
                item_id=f"{lat}_{lon}",
                start=lat_lon_precip[11323:].index[0],
                target=np.stack(
                    [
                        lat_lon_precip[11323:]["precip"].to_numpy(),
                        lat_lon_tmp2m["tmp2m"].to_numpy(),
                        lat_lon_tmin["tmin"].to_numpy(),
                        lat_lon_tmax["tmax"].to_numpy(),
                    ],
                    axis=0,
                ),
                freq="D",
            )

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            freq=Value("string"),
            target=Sequence(Sequence(Value("float32")), length=4),
        )
    )

    return gen_func, features


class SubseasonalDatasetBuilder(LOTSADatasetBuilder):
    dataset_list = [
        "subseasonal",
        "subseasonal_precip",
    ]
    dataset_type_map = defaultdict(lambda: MultiSampleTimeSeriesDataset)
    dataset_load_func_map = defaultdict(
        lambda: partial(
            MultiSampleTimeSeriesDataset, max_ts=128, combine_fields=("target",)
        )
    )

    def build_dataset(self, dataset: str):
        gen_func, features = {
            "subseasonal": _get_subseasonal_gen_func,
            "subseasonal_precip": _get_subseasonal_precip_gen_func,
        }[dataset]()

        hf_dataset = datasets.Dataset.from_generator(
            gen_func,
            features=features,
            cache_dir=env.HF_CACHE_PATH,
        )
        hf_dataset.info.dataset_name = dataset
        hf_dataset.save_to_disk(self.storage_path / dataset)
