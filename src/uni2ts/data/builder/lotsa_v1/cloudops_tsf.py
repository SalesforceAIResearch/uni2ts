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
from typing import Any, Generator

import datasets
from datasets import Features, Sequence, Value, load_dataset, load_dataset_builder
from gluonts.dataset.common import ProcessDataEntry
from gluonts.dataset.split import DateSplitter

from uni2ts.common.env import env
from uni2ts.data.dataset import TimeSeriesDataset

from ._base import LOTSADatasetBuilder


class CloudOpsTSFDatasetBuilder(LOTSADatasetBuilder):
    dataset_list = [
        "azure_vm_traces_2017",
        "borg_cluster_data_2011",
        "alibaba_cluster_trace_2018",
    ]
    dataset_type_map = defaultdict(lambda: TimeSeriesDataset)
    dataset_load_func_map = defaultdict(lambda: partial(TimeSeriesDataset))

    def build_dataset(self, dataset: str, num_proc: int = os.cpu_count()):
        cloudops_dataset = load_dataset(
            path="Salesforce/cloudops_tsf", name=dataset, split="pretrain"
        )
        cfg = load_dataset_builder(
            path="Salesforce/cloudops_tsf",
            name=dataset,
        ).config
        pde = ProcessDataEntry(
            freq=cfg.freq, one_dim_target=cfg.univariate, use_timestamp=False
        )
        splitter = DateSplitter(cfg.test_split_date)

        def process(entry):
            return next(iter(splitter.split([pde(entry)])[0]))

        def gen_func(ids: list[int]) -> Generator[dict[str, Any], None, None]:
            for item in cloudops_dataset.select(ids):
                item = process(item)
                yield dict(
                    item_id=item["item_id"],
                    start=item["start"].to_timestamp(),
                    freq=cfg.freq,
                    target=item["target"],
                    past_feat_dynamic_real=item["past_feat_dynamic_real"],
                )

        target_feature = (
            Sequence(Value("float32"))
            if cfg.target_dim == 1
            else Sequence(Sequence(Value("float32")), length=cfg.target_dim)
        )
        past_feat_dynamic_real_feature = Sequence(
            Sequence(Value("float32")), length=cfg.past_feat_dynamic_real_dim
        )

        hf_dataset = datasets.Dataset.from_generator(
            gen_func,
            features=Features(
                dict(
                    item_id=Value("string"),
                    start=Value("timestamp[s]"),
                    freq=Value("string"),
                    target=target_feature,
                    past_feat_dynamic_real=past_feat_dynamic_real_feature,
                )
            ),
            gen_kwargs={"ids": [i for i in range(len(cloudops_dataset))]},
            num_proc=num_proc,
            cache_dir=env.HF_CACHE_PATH,
        )
        hf_dataset.info.dataset_name = dataset
        hf_dataset.save_to_disk(
            self.storage_path / dataset,
            num_proc=10,
        )
