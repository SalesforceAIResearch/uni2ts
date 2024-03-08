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
from pathlib import Path
from typing import Any, Generator

import datasets
import pandas as pd
from datasets import Features, Sequence, Value

from uni2ts.common.env import env
from uni2ts.data.dataset import TimeSeriesDataset

from ._base import LOTSADatasetBuilder


class LargeSTDatasetBuilder(LOTSADatasetBuilder):
    dataset_list = [
        "largest_2017",
        "largest_2018",
        "largest_2019",
        "largest_2020",
        "largest_2021",
    ]
    dataset_type_map = defaultdict(lambda: TimeSeriesDataset)
    dataset_load_func_map = defaultdict(lambda: partial(TimeSeriesDataset))

    def build_dataset(self, dataset: str, num_proc: int = os.cpu_count()):
        year = dataset.split("_")[-1]
        df = pd.read_hdf(Path(os.getenv("LARGEST_PATH")) / f"ca_his_raw_{year}.h5")

        def gen_func(cols: list[int]) -> Generator[dict[str, Any], None, None]:
            for col in cols:
                if df[col].isnull().all():
                    continue
                yield dict(
                    item_id=f"{col}",
                    start=df.index[0],
                    target=df[col],
                    freq="5T",
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
            num_proc=num_proc,
            gen_kwargs={"cols": list(df.columns)},
            cache_dir=env.HF_CACHE_PATH,
        )
        hf_dataset.info.dataset_name = dataset
        hf_dataset.save_to_disk(self.storage_path / dataset)
