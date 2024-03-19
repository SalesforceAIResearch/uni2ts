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

import json
import os
from collections import defaultdict
from functools import partial
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
from datasets import Features, Sequence, Value
from pandas.tseries.frequencies import to_offset

from uni2ts.common.env import env
from uni2ts.data.dataset import MultiSampleTimeSeriesDataset

from ._base import LOTSADatasetBuilder


class LibCityDatasetBuilder(LOTSADatasetBuilder):
    dataset_list = [
        "BEIJING_SUBWAY_30MIN",
        "HZMETRO",
        "LOOP_SEATTLE",
        "LOS_LOOP",
        "M_DENSE",
        "PEMS03",
        "PEMS04",
        "PEMS07",
        "PEMS08",
        "PEMS_BAY",
        "Q-TRAFFIC",
        "SHMETRO",
        "SZ_TAXI",
    ]
    dataset_type_map = defaultdict(lambda: MultiSampleTimeSeriesDataset)
    dataset_load_func_map = defaultdict(
        lambda: partial(
            MultiSampleTimeSeriesDataset,
            max_ts=128,
            combine_fields=("target", "past_feat_dynamic_real"),
        )
    )

    def build_dataset(self, dataset: str, num_proc: int = os.cpu_count()):
        lib_city_path = Path(os.getenv("LIB_CITY_PATH"))
        engine = "pyarrow" if dataset == "Q-TRAFFIC" else None

        if dataset in ("PEMS03", "PEMS04", "PEMS07", "PEMS08"):
            raw_dataset_name = dataset.replace("0", "D")
        else:
            raw_dataset_name = dataset

        with open(lib_city_path / f"{raw_dataset_name}/config.json") as f:
            config = json.load(f)
        data_col = (
            data_col
            if len(data_col := config["info"]["data_col"]) == 1
            else config["info"]["data_col"]
        )
        freq = to_offset(
            pd.to_timedelta(f'{config["info"]["time_intervals"]}S')
        ).freqstr

        try:
            df = pd.read_csv(
                lib_city_path / f"{raw_dataset_name}/{raw_dataset_name}_new.dyna",
                engine=engine,
            )
        except FileNotFoundError:
            df = pd.read_csv(
                lib_city_path / f"{raw_dataset_name}/{raw_dataset_name}.dyna",
                engine=engine,
            )

        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time")

        past_feat_dynamic_real_dict = {}
        if (lib_city_path / f"{dataset}/{dataset}.ext").exists():
            ext_df = pd.read_csv(lib_city_path / f"{dataset}/{dataset}.ext")
            ext_df["time"] = pd.to_datetime(ext_df["time"])
            ext_df = ext_df.set_index("time")
            ext_df = ext_df[config["info"]["ext_col"]]
            if pd.infer_freq(ext_df.index) is None:
                ext_df = ext_df.reindex(
                    pd.date_range(ext_df.index[0], ext_df.index[-1], freq=freq)
                )
            cov = ext_df.to_numpy()
            if cov.shape[-1] == 1:
                cov = cov.squeeze(-1)
            else:
                cov = cov.T
            past_feat_dynamic_real_dict["past_feat_dynamic_real"] = cov
        else:
            cov = None

        entity_df = df[df.entity_id == next(iter(df.entity_id.unique()))]
        if entity_df[data_col].to_numpy().ndim == 2:
            target_dim = entity_df[data_col].to_numpy().astype(np.float32).T.shape[0]
            target_feature = (
                Sequence(Value("float32"))
                if target_dim == 1
                else Sequence(Sequence(Value("float32")), length=target_dim)
            )
        else:
            target_feature = Sequence(Value("float32"))

        past_feat_dynamic_real_feature_dict = {}
        if cov is not None:
            past_feat_dynamic_real_feature_dict["past_feat_dynamic_real"] = (
                Sequence(Value("float"))
                if cov.ndim == 1
                else Sequence(Sequence(Value("float32")), length=cov.shape[0])
            )

        def gen_func():
            for idx in df.entity_id.unique():
                entity_df = df.query(f"entity_id == {idx}")
                inferred_freq = pd.infer_freq(entity_df.index)
                if inferred_freq is None:
                    entity_df = entity_df.reindex(
                        pd.date_range(
                            entity_df.index[0], entity_df.index[-1], freq=freq
                        )
                    )
                target = entity_df[data_col].to_numpy().astype(np.float32)
                if target.ndim == 2:
                    if target.shape[-1] == 1:
                        target = target.squeeze(-1)
                    else:
                        target = target.T
                yield dict(
                    item_id=f"{idx}",
                    start=entity_df.index[0],
                    target=target,
                    freq=freq,
                ) | past_feat_dynamic_real_dict

        hf_datasets = datasets.Dataset.from_generator(
            gen_func,
            features=Features(
                dict(
                    item_id=Value("string"),
                    start=Value("timestamp[s]"),
                    freq=Value("string"),
                    target=target_feature,
                )
                | past_feat_dynamic_real_feature_dict
            ),
            cache_dir=env.HF_CACHE_PATH,
        )
        hf_datasets.info.dataset_name = dataset
        hf_datasets.save_to_disk(self.storage_path / dataset)
