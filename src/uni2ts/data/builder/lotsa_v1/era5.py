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

import itertools
import os
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Any, Generator

import datasets
import numpy as np
import pandas as pd
from datasets import Features, Sequence, Value

from uni2ts.common.env import env
from uni2ts.data.dataset import TimeSeriesDataset

from ._base import LOTSADatasetBuilder

ERA5_VARIABLES = [
    "2m_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
] + [
    f"{var}_{level}"
    for var, level in itertools.product(
        [
            "geopotential",
            "relative_humidity",
            "specific_humidity",
            "temperature",
            "u_component_of_wind",
            "v_component_of_wind",
        ],
        [50, 250, 500, 600, 700, 850, 925],
    )
]


class ERA5DatasetBuilder(LOTSADatasetBuilder):
    dataset_list = [f"era5_{year}" for year in range(1989, 2018 + 1)]
    dataset_type_map = defaultdict(lambda: TimeSeriesDataset)
    dataset_load_func_map = defaultdict(lambda: partial(TimeSeriesDataset))
    uniform = True

    def build_dataset(self, dataset: str, num_proc: int = os.cpu_count()):
        era5_path = Path(os.getenv("ERA5_PATH"))

        year = dataset.split("_")[-1]
        all_jobs = [(x, y) for x, y in itertools.product(range(64), range(128))]

        all_vars = {var: [] for var in ERA5_VARIABLES}
        for shard in range(16):
            np_file = np.load(str(era5_path / f"train/{year}_{shard}.npz"))
            for var in ERA5_VARIABLES:
                all_vars[var].append(np_file[var][:, 0, :, :])

        targets = np.stack(
            [np.concatenate(all_vars[var]) for var in ERA5_VARIABLES], axis=0
        )

        def gen_func(
            jobs: list[tuple[int, int]]
        ) -> Generator[dict[str, Any], None, None]:
            for x, y in jobs:
                yield dict(
                    item_id=f"{year}_{x}_{y}",
                    start=pd.Timestamp(f"{year}-01-01"),
                    target=targets[:, :, x, y],
                    freq="H",
                )

        hf_dataset = datasets.Dataset.from_generator(
            gen_func,
            features=Features(
                dict(
                    item_id=Value("string"),
                    start=Value("timestamp[s]"),
                    freq=Value("string"),
                    target=Sequence(
                        Sequence(Value("float32")), length=len(ERA5_VARIABLES)
                    ),
                )
            ),
            gen_kwargs=dict(jobs=all_jobs),
            num_proc=num_proc,
            cache_dir=env.HF_CACHE_PATH,
        )
        hf_dataset.info.dataset_name = dataset
        hf_dataset.save_to_disk(
            self.storage_path / dataset,
            num_proc=num_proc,
        )
