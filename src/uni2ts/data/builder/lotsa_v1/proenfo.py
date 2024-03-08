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

import datasets
import numpy as np
import pandas as pd
from datasets import Features, Sequence, Value

from uni2ts.common.env import env
from uni2ts.common.typing import GenFunc
from uni2ts.data.dataset import MultiSampleTimeSeriesDataset, TimeSeriesDataset

from ._base import LOTSADatasetBuilder

MULTI_SAMPLE_DATASETS = [
    "gfc12_load",
    "gfc17_load",
    "bull",
    "hog",
]


def _get_gfc12_load_gen_func(proenfo_path: Path) -> tuple[GenFunc, Features]:
    df = pd.read_pickle(proenfo_path / "GFC12_load/new_load_with_weather.pkl")

    def gen_func():
        for col in range(1, 21):
            yield dict(
                item_id=f"GFC12_load_{col}",
                start=df.index[0],
                target=df[str(col)],
                past_feat_dynamic_real=df.airTemperature,
                freq="H",
            )

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            target=Sequence(Value("float32")),
            past_feat_dynamic_real=Sequence(Value("float32")),
            freq=Value("string"),
        )
    )

    return gen_func, features


def _get_gfc14_load_gen_func(proenfo_path: Path) -> tuple[GenFunc, Features]:
    df = pd.read_pickle(proenfo_path / "GFC14_load/load_with_weather.pkl")

    def gen_func():
        yield dict(
            item_id="GFC14_load",
            start=df.index[0],
            target=df["load"],
            past_feat_dynamic_real=df.airTemperature,
            freq="H",
        )

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            target=Sequence(Value("float32")),
            freq=Value("string"),
        )
    )

    return gen_func, features


def _get_gfc17_load_gen_func(proenfo_path: Path) -> tuple[GenFunc, Features]:
    df = pd.read_pickle(proenfo_path / "GFC17_load/load_with_weather.pkl")

    def gen_func():
        for col in ["CT", "ME", "NH", "RI", "VT", "NEMASSBOST", "SEMASS", "WCMASS"]:
            yield dict(
                item_id=f"GFC17_load_{col}",
                start=df[col].index[0],
                target=df[col]["load"].astype(np.float32),
                past_feat_dynamic_real=df[col]["airTemperature"].astype(np.float32),
                freq="H",
            )

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            target=Sequence(Value("float32")),
            past_feat_dynamic_real=Sequence(Value("float32")),
            freq=Value("string"),
        )
    )

    return gen_func, features


def _get_spain_gen_func(proenfo_path: Path) -> tuple[GenFunc, Features]:
    df = pd.read_pickle(proenfo_path / "Spain/new_load_with_weather.pkl")

    def gen_func():
        yield dict(
            item_id="Spain",
            start=df.index[0],
            target=df["load"],
            past_feat_dynamic_real=df.airTemperature,
            freq="H",
        )

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            target=Sequence(Value("float32")),
            past_feat_dynamic_real=Sequence(Value("float32")),
            freq=Value("string"),
        )
    )

    return gen_func, features


def _get_pdb_gen_func(proenfo_path: Path) -> tuple[GenFunc, Features]:
    df = pd.read_pickle(proenfo_path / "PDB/load_with_weather.pkl")

    def gen_func():
        yield dict(
            item_id="PDB",
            start=df.index[0],
            target=df["load"].astype(np.float32),
            past_feat_dynamic_real=df.airTemperature.astype(np.float32),
            freq="H",
        )

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            target=Sequence(Value("float32")),
            past_feat_dynamic_real=Sequence(Value("float32")),
            freq=Value("string"),
        )
    )

    return gen_func, features


def _get_elf_gen_func(proenfo_path: Path):
    df = pd.read_pickle(proenfo_path / "ELF/load_with_weather.pkl")

    def gen_func():
        yield dict(
            item_id="ELF",
            start=df.index[0],
            target=df["load"],
            freq="H",
        )

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            target=Sequence(Value("float32")),
            freq=Value("string"),
        )
    )

    return gen_func, features


def _get_bull_gen_func(proenfo_path: Path) -> tuple[GenFunc, Features]:
    df = pd.read_pickle(proenfo_path / "Bull/load_with_weather.pkl")

    def gen_func():
        for col in [col for col in df if "Bull" in col]:
            yield dict(
                item_id=f"Bull_{col}",
                start=df.index[0],
                target=df[col],
                past_feat_dynamic_real=df[
                    ["airTemperature", "dewTemperature", "seaLvlPressure"]
                ]
                .to_numpy()
                .T,
                freq="H",
            )

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            target=Sequence(Value("float32")),
            past_feat_dynamic_real=Sequence(Sequence(Value("float32")), length=3),
            freq=Value("string"),
        )
    )

    return gen_func, features


def _get_cockatoo_gen_func(proenfo_path: Path) -> tuple[GenFunc, Features]:
    df = pd.read_pickle(proenfo_path / "Cockatoo/load_with_weather.pkl")

    def gen_func():
        for col in [col for col in df if "Cockatoo" in col]:
            yield dict(
                item_id=f"Cockatoo_{col}",
                start=df.index[0],
                target=df[col],
                past_feat_dynamic_real=df[
                    [
                        "airTemperature",
                        "dewTemperature",
                        "seaLvlPressure",
                        "windDirection",
                        "windSpeed",
                    ]
                ]
                .to_numpy()
                .T,
                freq="H",
            )

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            target=Sequence(Value("float32")),
            past_feat_dynamic_real=Sequence(Sequence(Value("float32")), length=5),
            freq=Value("string"),
        )
    )

    return gen_func, features


def _get_hog_gen_func(proenfo_path: Path) -> tuple[GenFunc, Features]:
    df = pd.read_pickle(proenfo_path / "Hog/load_with_weather.pkl")

    def gen_func():
        for col in [col for col in df if "Hog" in col]:
            yield dict(
                item_id=f"Hog_{col}",
                start=df.index[0],
                target=df[col],
                past_feat_dynamic_real=df[
                    [
                        "airTemperature",
                        "dewTemperature",
                        "seaLvlPressure",
                        "windDirection",
                        "windSpeed",
                    ]
                ]
                .to_numpy()
                .T,
                freq="H",
            )

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            target=Sequence(Value("float32")),
            past_feat_dynamic_real=Sequence(Sequence(Value("float32")), length=5),
            freq=Value("string"),
        )
    )

    return gen_func, features


def _get_covid19_energy_gen_func(proenfo_path: Path) -> tuple[GenFunc, Features]:
    df = pd.read_pickle(proenfo_path / "Covid19/load_with_weather.pkl")
    cols = list(df.columns)
    cols.remove("load")
    target = df["load"]
    past_feat_dynamic_real = df[cols].to_numpy().T

    def gen_func():
        yield dict(
            item_id="covid19_energy",
            start=df.index[0],
            target=target,
            past_feat_dynamic_real=past_feat_dynamic_real,
            freq="H",
        )

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            target=Sequence(Value("float32")),
            past_feat_dynamic_real=Sequence(
                Sequence(Value("float32")), length=len(past_feat_dynamic_real)
            ),
            freq=Value("string"),
        )
    )

    return gen_func, features


class ProEnFoDatasetBuilder(LOTSADatasetBuilder):
    dataset_list = [
        "gfc12_load",
        "gfc14_load",
        "gfc17_load",
        "spain",
        "pdb",
        "elf",
        "bull",
        "cockatoo",
        "hog",
        "covid19_energy",
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
        proenfo_path = Path(os.getenv("PROENFO_PATH"))
        gen_func, features = {
            "gfc12_load": _get_gfc12_load_gen_func,
            "gfc14_load": _get_gfc14_load_gen_func,
            "gfc17_load": _get_gfc17_load_gen_func,
            "spain": _get_spain_gen_func,
            "pdb": _get_pdb_gen_func,
            "elf": _get_elf_gen_func,
            "bull": _get_bull_gen_func,
            "cockatoo": _get_cockatoo_gen_func,
            "hog": _get_hog_gen_func,
            "covid19_energy": _get_covid19_energy_gen_func,
        }[dataset](proenfo_path)

        hf_dataset = datasets.Dataset.from_generator(
            gen_func,
            features=features,
            cache_dir=env.HF_CACHE_PATH,
        )
        hf_dataset.info.dataset_name = dataset
        hf_dataset.save_to_disk(self.storage_path / dataset)
