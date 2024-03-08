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

import os
from collections import defaultdict
from functools import partial
from itertools import product
from pathlib import Path
from typing import Any, Generator

import datasets
import pyarrow.parquet as pq
from datasets import Features, Sequence, Value

try:
    from buildings_bench.data import load_pandas_dataset
except ImportError:
    import traceback

    e = traceback.format_exc()

    def load_pandas_dataset(*args, **kwargs):
        raise ImportError(e)


from uni2ts.common.env import env
from uni2ts.data.dataset import MultiSampleTimeSeriesDataset, TimeSeriesDataset

from ._base import LOTSADatasetBuilder

MULTI_SAMPLE_DATASETS = [
    "bdg-2_panther",
]


class BuildingsBenchDatasetBuilder(LOTSADatasetBuilder):
    dataset_list = [
        "sceaux",
        "borealis",
        "ideal",
        "bdg-2_panther",
        "bdg-2_fox",
        "bdg-2_rat",
        "bdg-2_bear",
        "smart",
        "lcl",
    ]
    dataset_type_map = defaultdict(lambda: TimeSeriesDataset) | {
        dataset: MultiSampleTimeSeriesDataset for dataset in MULTI_SAMPLE_DATASETS
    }
    dataset_load_func_map = defaultdict(lambda: partial(TimeSeriesDataset)) | {
        dataset: partial(
            MultiSampleTimeSeriesDataset,
            max_ts=128,
            combine_fields=("target", "past_feat_dynamic_real"),
        )
        for dataset in MULTI_SAMPLE_DATASETS
    }

    def build_dataset(self, dataset: str):
        def gen_func() -> Generator[dict[str, Any], None, None]:
            if dataset.startswith("bdg"):
                pd_dataset = load_pandas_dataset(dataset.replace("_", ":"))
            else:
                pd_dataset = load_pandas_dataset(dataset)

            for item_id, df in pd_dataset:
                freq = df.index.freqstr
                if freq is None:
                    from pandas.tseries.frequencies import to_offset

                    freq = to_offset(df.index[1] - df.index[0]).freqstr
                    df = df.asfreq(freq)
                    if df.power.isnull().sum() / len(df) > 0.5:
                        continue
                yield dict(
                    item_id=item_id,
                    start=df.index[0],
                    target=df.power,
                    freq=freq,
                )

        hf_dataset = datasets.Dataset.from_generator(
            generator=gen_func,
            features=Features(
                dict(
                    item_id=Value("string"),
                    start=Value("timestamp[s]"),
                    freq=Value("string"),
                    target=Sequence(Value("float32")),
                )
            ),
            cache_dir=env.HF_CACHE_PATH,
        )
        hf_dataset.info.dataset_name = dataset
        hf_dataset.save_to_disk(dataset_path=env.LOTSA_V1_PATH / dataset)


class Buildings900KDatasetBuilder(LOTSADatasetBuilder):
    dataset_list: list[str] = ["buildings_900k"]
    dataset_type_map = dict(buildings_900k=TimeSeriesDataset)
    dataset_load_func_map = dict(
        buildings_900k=partial(TimeSeriesDataset),
    )

    def build_dataset(self, dataset: str, num_proc: int = os.cpu_count()):
        all_jobs = []
        building_type_and_years = [
            "comstock_amy2018",
            "comstock_tmy3",
            "resstock_amy2018",
            "resstock_tmy3",
        ]
        regions = ["midwest", "northeast", "south", "west"]
        for building_type_and_year, region in product(building_type_and_years, regions):
            for building_dir in [
                "Buildings-900K/end-use-load-profiles-for-us-building-stock",
                "Buildings-900K-test",
            ]:
                pumas = (
                    Path(os.getenv("BUILDINGS_BENCH"))
                    / f"{building_dir}/2021/{building_type_and_year}_release_1/"
                    f"timeseries_individual_buildings/by_puma_{region}/upgrade=0/"
                ).glob("puma=*")

                pumas = [p.stem[len("puma=") :] for p in pumas]

                for puma in pumas:
                    all_jobs.append(
                        (building_dir, building_type_and_year, region, puma)
                    )

        def gen_func(job_ids: list[int]) -> Generator[dict[str, Any], None, None]:
            for idx in job_ids:
                building_dir, building_type_and_year, region, puma = all_jobs[idx]
                tab = pq.read_table(
                    Path(os.getenv("BUILDINGS_BENCH"))
                    / f"{building_dir}/2021/{building_type_and_year}_release_1/"
                    f"timeseries_individual_buildings/by_puma_{region}/upgrade=0/puma={puma}"
                )
                tab = tab.sort_by("timestamp")
                for building_num, col_num in zip(
                    tab.column_names[1:], range(1, tab.num_columns)
                ):
                    yield dict(
                        item_id=f"{building_type_and_year}_{region}_{puma}_{building_num}",
                        start=tab.column(0)
                        .slice(0, 1)
                        .to_numpy()[0]
                        .astype("datetime64"),
                        target=tab.column(col_num).to_numpy(),
                        freq="H",
                    )

        hf_dataset = datasets.Dataset.from_generator(
            gen_func,
            features=Features(
                dict(
                    item_id=Value("string"),
                    start=Value("timestamp[s]"),
                    freq=Value("string"),
                    target=Sequence(Value("float32")),
                )
            ),
            gen_kwargs={"job_ids": [i for i in range(len(all_jobs))]},
            num_proc=num_proc,
            cache_dir=env.HF_CACHE_PATH,
        )
        hf_dataset.info.dataset_name = dataset
        hf_dataset.save_to_disk(self.storage_path / dataset, num_proc=num_proc)
