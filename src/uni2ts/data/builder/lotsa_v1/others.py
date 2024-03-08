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
import numpy as np
import pandas as pd
from datasets import Features, Sequence, Value

try:
    from pyreadr import read_r
except ImportError as e:

    def read_r(*args, **kwargs):
        raise e


from uni2ts.common.env import env
from uni2ts.common.typing import GenFunc
from uni2ts.data.dataset import MultiSampleTimeSeriesDataset, TimeSeriesDataset

from ._base import LOTSADatasetBuilder

MULTI_SAMPLE_DATASETS = [
    "godaddy",
    "hierarchical_sales",
    "beijing_air_quality",
]


def _get_kdd_2022_gen_func(data_path: Path) -> tuple[GenFunc, Features]:
    df = pd.read_csv(
        data_path / "wtbdata_245days.csv", header=0, skiprows=lambda x: x == 1
    )
    df["timestamp"] = pd.to_datetime("2020-01-01 " + df["Tmstamp"]) + (
        df["Day"] - 1
    ) * pd.Timedelta("1D")

    def gen_func() -> Generator[dict[str, Any], None, None]:
        for idx in df["TurbID"].unique():
            id_df = df[df["TurbID"] == idx].set_index("timestamp")
            yield dict(
                item_id=f"{idx}",
                start=id_df.index[0],
                target=id_df["Patv"],
                past_feat_dynamic_real=id_df[
                    [
                        "Wspd",
                        "Wdir",
                        "Etmp",
                        "Itmp",
                        "Ndir",
                        "Pab1",
                        "Pab2",
                        "Pab3",
                        "Prtv",
                    ]
                ]
                .to_numpy()
                .T,
                freq="10T",
            )

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            target=Sequence(Value("float32")),
            past_feat_dynamic_real=Sequence(Sequence(Value("float32")), length=9),
            freq=Value("string"),
        )
    )

    return gen_func, features


def _get_godaddy_gen_func(data_path: Path) -> tuple[GenFunc, Features]:
    train = pd.read_csv(data_path / "train.csv")
    test = pd.read_csv(data_path / "revealed_test.csv")
    df = pd.concat([train, test])
    df["first_day_of_month"] = pd.to_datetime(df["first_day_of_month"])

    def gen_func() -> Generator[dict[str, Any], None, None]:
        for idx in df.cfips.unique():
            id_df = df.query(f"cfips == {idx}")
            id_df = id_df.set_index("first_day_of_month").sort_index()
            yield dict(
                item_id=f"{idx}",
                start=id_df.index[0],
                target=id_df[["microbusiness_density", "active"]].to_numpy().T,
                freq="MS",
            )

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            target=Sequence(Sequence(Value("float32")), length=2),
            freq=Value("string"),
        )
    )

    return gen_func, features


def _get_favorita_sales_gen_func(
    data_path: Path, length_threshold: int = 250, missing_threshold: float = 0.5
) -> tuple[GenFunc, Features]:
    train = pd.read_csv(
        data_path / "train.csv",
        dtype=dict(
            id=int, store_nbr=int, item_nbr=int, unit_sales=int, onpromotion=bool
        ),
        parse_dates=["date"],
        engine="pyarrow",
    )

    def gen_func() -> Generator[dict[str, Any], None, None]:
        for store_nbr in train.store_nbr.unique():
            store = train.query(f"store_nbr == {store_nbr}")
            for item_nbr in store.item_nbr.unique():
                item = (
                    store.query(f"item_nbr == {item_nbr}")
                    .set_index("date")
                    .sort_index()
                )
                item = item.reindex(
                    pd.date_range(start=item.index[0], end=item.index[-1], freq="1D")
                )

                missing_pct = item.unit_sales.isnull().sum() / len(item)

                if len(item) < length_threshold or missing_pct > missing_threshold:
                    continue

                yield dict(
                    item_id=f"{store_nbr}_{item_nbr}",
                    start=item.index[0],
                    target=item.unit_sales,
                    freq="D",
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


def _get_favorita_transactions_gen_func(data_path: Path) -> tuple[GenFunc, Features]:
    transactions = pd.read_csv(data_path / "transactions.csv")
    transactions["date"] = pd.to_datetime(transactions["date"])

    def gen_func() -> Generator[dict[str, Any], None, None]:
        for store_nbr in transactions.store_nbr.unique():
            store = (
                transactions.query(f"store_nbr == {store_nbr}")
                .set_index("date")
                .sort_index()
            )
            store = store.reindex(
                pd.date_range(start=store.index[0], end=store.index[-1], freq="1D")
            )
            yield dict(
                item_id=f"{store_nbr}",
                start=store.index[0],
                target=store.transactions,
                freq="D",
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


def _get_restaurant_gen_func(
    data_path: Path, missing_threshold: float = 0.5
) -> tuple[GenFunc, Features]:
    air_visit_data = pd.read_csv(data_path / "air_visit_data.csv")
    air_visit_data["visit_date"] = pd.to_datetime(air_visit_data["visit_date"])

    def gen_func() -> Generator[dict[str, Any], None, None]:
        for air_store_id in air_visit_data.air_store_id.unique():
            air_store = (
                air_visit_data.query(f'air_store_id == "{air_store_id}"')
                .set_index("visit_date")
                .sort_index()
            )
            air_store = air_store.reindex(
                pd.date_range(
                    start=air_store.index[0], end=air_store.index[-1], freq="1D"
                )
            )
            missing_pct = air_store.visitors.isnull().sum() / len(air_store)
            if missing_pct > missing_threshold:
                continue
            yield dict(
                item_id=f"{air_store_id}",
                start=air_store.index[0],
                target=air_store.visitors,
                freq="D",
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


def _get_hierarchical_sales_gen_func(data_path: Path) -> tuple[GenFunc, Features]:
    sales = pd.read_csv(data_path / "hierarchical_sales_data.csv")
    sales["DATE"] = pd.to_datetime(sales["DATE"])
    sales = sales.set_index("DATE").sort_index()

    def gen_func() -> Generator[dict[str, Any], None, None]:
        for id in sales.columns:
            if "QTY" in id:
                yield dict(
                    item_id=id,
                    start=sales.index[0],
                    target=sales[id].astype(np.float32),
                    freq="D",
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


def _get_china_air_quality_gen_func(data_path: Path) -> tuple[GenFunc, Features]:
    airquality = pd.read_csv(data_path / "airquality.csv")
    airquality["time"] = pd.to_datetime(airquality["time"])

    def gen_func() -> Generator[dict[str, Any], None, None]:
        for station_id in airquality.station_id.unique():
            station = (
                airquality.query(f"station_id == {station_id}")
                .set_index("time")
                .sort_index()
            )
            station = station.reindex(
                pd.date_range(
                    start=station.index[0],
                    end=station.index[-1],
                    freq="H",
                )
            )
            yield dict(
                item_id=f"{station_id}",
                start=station.index[0],
                target=station[
                    [
                        "PM25_Concentration",
                        "PM10_Concentration",
                        "NO2_Concentration",
                        "CO_Concentration",
                        "O3_Concentration",
                        "SO2_Concentration",
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
            target=Sequence(Sequence(Value("float32")), length=6),
            freq=Value("string"),
        )
    )

    return gen_func, features


def _get_beijing_air_quality_gen_func(data_path: Path) -> tuple[GenFunc, Features]:
    files = data_path.glob("*.csv")
    dfs = [(file, pd.read_csv(file)) for file in files]

    def gen_func() -> Generator[dict[str, Any], None, None]:
        for file, data in dfs:
            data["date"] = pd.to_datetime(
                data.year.astype(str)
                + "-"
                + data.month.astype(str)
                + "-"
                + data.day.astype(str)
                + " "
                + data.hour.astype(str)
                + ":00"
            )
            data = data.set_index("date").sort_index()
            data = data.reindex(
                pd.date_range(
                    start=data.index[0],
                    end=data.index[-1],
                    freq="H",
                )
            )
            yield dict(
                item_id=file.stem,
                start=data.index[0],
                target=data[
                    [
                        "PM2.5",
                        "PM10",
                        "SO2",
                        "NO2",
                        "CO",
                        "O3",
                        "TEMP",
                        "PRES",
                        "DEWP",
                        "RAIN",
                        "WSPM",
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
            target=Sequence(Sequence(Value("float32")), length=11),
            freq=Value("string"),
        )
    )

    return gen_func, features


def _get_residential_load_power_gen_func(data_path: Path) -> tuple[GenFunc, Features]:
    units = [
        file.stem
        for file in (data_path / "anonymous_public_load_power_data_per_unit").glob(
            "*.rds"
        )
    ]

    def gen_func() -> Generator[dict[str, Any], None, None]:
        for unit in units:
            load_power = read_r(
                data_path / f"anonymous_public_load_power_data_per_unit/{unit}.rds"
            )[None]
            load_power = (
                load_power.drop_duplicates(subset="utc", keep="last")
                .set_index("utc")
                .sort_index()
            )
            load_power = load_power.reindex(
                pd.date_range(
                    start=load_power.index[0],
                    end=load_power.index[-1],
                    freq="T",
                )
            )
            target = load_power[["sum", "min", "max"]].to_numpy()
            if target.shape[0] < 16:
                continue

            yield dict(
                item_id=f"{unit}",
                start=load_power.index[0],
                target=target.T,
                freq="T",
            )

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            target=Sequence(Sequence(Value("float32")), length=3),
            freq=Value("string"),
        )
    )

    return gen_func, features


def _get_residential_pv_power_gen_func(data_path: Path) -> tuple[GenFunc, Features]:
    units = [
        file.stem
        for file in (data_path / "anonymous_public_pv_power_data_per_unit").glob(
            "*.rds"
        )
    ]

    def gen_func() -> Generator[dict[str, Any], None, None]:
        for unit in units:
            pv_power = read_r(
                data_path / f"anonymous_public_pv_power_data_per_unit/{unit}.rds"
            )[None]
            pv_power = (
                pv_power.drop_duplicates(subset="utc", keep="last")
                .set_index("utc")
                .sort_index()
            )
            pv_power = pv_power.reindex(
                pd.date_range(
                    start=pv_power.index[0],
                    end=pv_power.index[-1],
                    freq="T",
                )
            )
            yield dict(
                item_id=f"{unit}",
                start=pv_power.index[0],
                target=pv_power[["sum", "min", "max"]].to_numpy().T,
                freq="T",
            )

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            target=Sequence(Sequence(Value("float32")), length=3),
            freq=Value("string"),
        )
    )

    return gen_func, features


def _get_cdc_fluview_ilinet_gen_func(data_path: Path) -> tuple[GenFunc, Features]:
    national = pd.read_csv(data_path / "National/ILINet.csv", skiprows=1)
    hhs = pd.read_csv(data_path / "HHS/ILINet.csv", skiprows=1)
    census = pd.read_csv(data_path / "Census/ILINet.csv", skiprows=1)
    state = pd.read_csv(data_path / "State/ILINet.csv", skiprows=1)

    def gen_func() -> Generator[dict[str, Any], None, None]:
        for dataset in (national, hhs, census, state):
            dataset["date"] = pd.to_datetime(
                (dataset.YEAR * 100 + dataset.WEEK).astype(str) + "0", format="%Y%W%w"
            )
            for region in dataset.REGION.unique():
                region_ds = (
                    dataset.query(f'REGION == "{region}"')
                    .set_index("date")
                    .sort_index()
                )

                if region_ds["REGION TYPE"].iloc[0] == "National":
                    item_id = "national"
                elif region_ds["REGION TYPE"].iloc[0] == "HHS Regions":
                    item_id = f"hhs_{region}"
                elif region_ds["REGION TYPE"].iloc[0] == "Census Regions":
                    item_id = f"census_{region}"
                elif region_ds["REGION TYPE"].iloc[0] == "States":
                    item_id = f"states_{region}"
                else:
                    raise ValueError

                yield dict(
                    item_id=item_id,
                    start=region_ds.index[0],
                    target=region_ds[
                        [
                            "% WEIGHTED ILI",
                            "%UNWEIGHTED ILI",
                            "ILITOTAL",
                            "NUM. OF PROVIDERS",
                            "TOTAL PATIENTS",
                        ]
                    ]
                    .replace("X", np.nan)
                    .to_numpy()
                    .astype(float)
                    .T,
                    freq="W",
                )

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            target=Sequence(Sequence(Value("float32")), length=5),
            freq=Value("string"),
        )
    )

    return gen_func, features


def _get_cdc_fluview_who_nrevss_gen_func(data_path: Path) -> tuple[GenFunc, Features]:
    national_prior = pd.read_csv(
        data_path / "National/WHO_NREVSS_Combined_prior_to_2015_16.csv", skiprows=1
    )
    national_public_health = pd.read_csv(
        data_path / "National/WHO_NREVSS_Public_Health_Labs.csv", skiprows=1
    )
    national_clinical_labs = pd.read_csv(
        data_path / "National/WHO_NREVSS_Clinical_Labs.csv", skiprows=1
    )
    hhs_prior = pd.read_csv(
        data_path / "HHS/WHO_NREVSS_Combined_prior_to_2015_16.csv", skiprows=1
    )
    hhs_public_health = pd.read_csv(
        data_path / "HHS/WHO_NREVSS_Public_Health_Labs.csv", skiprows=1
    )
    hhs_clinical_labs = pd.read_csv(
        data_path / "HHS/WHO_NREVSS_Clinical_Labs.csv", skiprows=1
    )
    census_prior = pd.read_csv(
        data_path / "Census/WHO_NREVSS_Combined_prior_to_2015_16.csv", skiprows=1
    )
    census_public_health = pd.read_csv(
        data_path / "Census/WHO_NREVSS_Public_Health_Labs.csv", skiprows=1
    )
    census_clinical_labs = pd.read_csv(
        data_path / "Census/WHO_NREVSS_Clinical_Labs.csv", skiprows=1
    )
    state_prior = pd.read_csv(
        data_path / "State/WHO_NREVSS_Combined_prior_to_2015_16.csv", skiprows=1
    )
    state_public_health = pd.read_csv(
        data_path / "State/WHO_NREVSS_Public_Health_Labs.csv", skiprows=1
    )
    state_clinical_labs = pd.read_csv(
        data_path / "State/WHO_NREVSS_Clinical_Labs.csv", skiprows=1
    )

    state_public_health["YEAR"] = (
        state_public_health["SEASON_DESCRIPTION"]
        .apply(lambda x: x[len("Season ") : len("Season 2015")])
        .astype(int)
    )
    state_public_health["WEEK"] = (
        state_public_health["SEASON_DESCRIPTION"]
        .apply(lambda x: x[len("Season 2015-") :])
        .astype(int)
    )

    def gen_func() -> Generator[dict[str, Any], None, None]:
        for prior, public_health, clinical_labs in [
            (national_prior, national_public_health, national_clinical_labs),
            (hhs_prior, hhs_public_health, hhs_clinical_labs),
            (census_prior, census_public_health, census_clinical_labs),
            (state_prior, state_public_health, state_clinical_labs),
        ]:
            for col in [
                "TOTAL SPECIMENS",
                "A (2009 H1N1)",
                "A (H1)",
                "A (H3)",
                "A (Subtyping not Performed)",
                "A (Unable to Subtype)",
                "B",
                "H3N2v",
            ]:
                prior[col] = prior[col].replace("X", 0).astype(int)

            for col in [
                "TOTAL SPECIMENS",
                "A (2009 H1N1)",
                "A (H3)",
                "A (Subtyping not Performed)",
                "B",
                "BVic",
                "BYam",
                "H3N2v",
            ]:
                public_health[col] = public_health[col].replace("X", 0).astype(int)

            for col in ["TOTAL SPECIMENS", "TOTAL A", "TOTAL B"]:
                clinical_labs[col] = clinical_labs[col].replace("X", 0).astype(int)

            prior.loc[:, "A"] = (
                prior["A (2009 H1N1)"]
                + prior["A (H1)"]
                + prior["A (H3)"]
                + prior["A (Subtyping not Performed)"]
                + prior["A (Unable to Subtype)"]
            )
            public_health.loc[:, "A"] = (
                public_health["A (2009 H1N1)"]
                + public_health["A (H3)"]
                + public_health["A (Subtyping not Performed)"]
            )
            public_health.loc[:, "B"] = (
                public_health["B"] + public_health["BVic"] + public_health["BYam"]
            )

            prior = prior[
                [
                    "TOTAL SPECIMENS",
                    "A",
                    "B",
                    "H3N2v",
                    "YEAR",
                    "WEEK",
                    "REGION",
                    "REGION TYPE",
                ]
            ]
            post = public_health[
                [
                    "TOTAL SPECIMENS",
                    "A",
                    "B",
                    "H3N2v",
                    "YEAR",
                    "WEEK",
                    "REGION",
                    "REGION TYPE",
                ]
            ]
            post.loc[:, "TOTAL SPECIMENS"] = (
                post["TOTAL SPECIMENS"] + clinical_labs["TOTAL SPECIMENS"]
            )
            post.loc[:, "A"] = post["A"] + clinical_labs["TOTAL A"]
            post.loc[:, "B"] = post["B"] + clinical_labs["TOTAL B"]

            combined = pd.concat([prior, post])
            combined["date"] = pd.to_datetime(
                (combined.YEAR * 100 + combined.WEEK).astype(str) + "0", format="%Y%W%w"
            )

            for region in combined.REGION.unique():
                region_ds = (
                    combined.query(f'REGION == "{region}"')
                    .set_index("date")
                    .sort_index()
                )

                if region_ds["REGION TYPE"].iloc[0] == "National":
                    item_id = "national"
                elif region_ds["REGION TYPE"].iloc[0] == "HHS Regions":
                    item_id = f"hhs_{region}"
                elif region_ds["REGION TYPE"].iloc[0] == "Census Regions":
                    item_id = f"census_{region}"
                elif region_ds["REGION TYPE"].iloc[0] == "States":
                    item_id = f"states_{region}"
                else:
                    raise ValueError

                target = (
                    region_ds[["TOTAL SPECIMENS", "A", "B", "H3N2v"]]
                    .to_numpy()
                    .astype(np.float32)
                )

                if target.shape[0] < 16:
                    continue

                yield dict(
                    item_id=item_id,
                    start=region_ds.index[0],
                    target=target.T,
                    freq="W",
                )

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            target=Sequence(Sequence(Value("float32")), length=4),
            freq=Value("string"),
        )
    )

    return gen_func, features


def _get_project_tycho_gen_func(
    data_path: Path,
    length_threshold: int = 100,
    missing_threshold: float = 0.25,
) -> tuple[GenFunc, Features]:
    data = pd.read_csv(data_path / "ProjectTycho_Level2_v1.1.0.csv", engine="pyarrow")
    data = data.rename(columns={" event": "event"})
    ts = data[["state", "loc", "loc_type", "disease", "event"]].value_counts()
    ts = ts[ts > 0].index

    def gen_func() -> Generator[dict[str, Any], None, None]:
        for state, loc, loc_type, disease, event in ts:
            item = data.query(
                f'state == "{state}" and loc == "{loc}" and loc_type == "{loc_type}" and '
                f'disease == "{disease}" and event == "{event}"'
            )
            item.loc[:, "from_date"] = pd.to_datetime(item["from_date"])
            item = item.drop_duplicates("from_date").set_index("from_date").sort_index()

            item = item.reindex(
                pd.date_range(
                    start=item.index[0],
                    end=item.index[-1],
                    freq="W",
                )
            )

            missing_pct = item.number.isnull().sum() / len(item)
            if len(item) < length_threshold or missing_pct > missing_threshold:
                continue

            yield dict(
                item_id=f"{state}_{loc}_{loc_type}_{disease}_{event}",
                start=item.index[0],
                target=item.number,
                freq="W",
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


class OthersLOTSADatasetBuilder(LOTSADatasetBuilder):
    dataset_list = [
        "kdd2022",
        "godaddy",
        "favorita_sales",
        "favorita_transactions",
        "restaurant",
        "hierarchical_sales",
        "china_air_quality",
        "beijing_air_quality",
        "residential_load_power",
        "residential_pv_power",
        "cdc_fluview_ilinet",
        "cdc_fluview_who_nrevss",
        "project_tycho",
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
        data_path = (
            Path(os.getenv("OTHERS_PATH"))
            / {
                "kdd2022": "kdd2022",
                "godaddy": "godaddy",
                "favorita_sales": "favorita",
                "favorita_transactions": "favorita",
                "restaurant": "restaurant",
                "hierarchical_sales": "hierarchical_sales",
                "china_air_quality": "china_air_quality",
                "beijing_air_quality": "beijing_air_quality",
                "residential_load_power": "residential_power",
                "residential_pv_power": "residential_power",
                "cdc_fluview_ilinet": "CDCFluView",
                "cdc_fluview_who_nrevss": "CDCFluView",
                "project_tycho": "ProjectTycho",
            }[dataset]
        )
        gen_func, features = {
            "kdd2022": _get_kdd_2022_gen_func,
            "godaddy": _get_godaddy_gen_func,
            "favorita_sales": _get_favorita_sales_gen_func,
            "favorita_transactions": _get_favorita_transactions_gen_func,
            "restaurant": _get_restaurant_gen_func,
            "hierarchical_sales": _get_hierarchical_sales_gen_func,
            "china_air_quality": _get_china_air_quality_gen_func,
            "beijing_air_quality": _get_beijing_air_quality_gen_func,
            "residential_load_power": _get_residential_load_power_gen_func,
            "residential_pv_power": _get_residential_pv_power_gen_func,
            "cdc_fluview_ilinet": _get_cdc_fluview_ilinet_gen_func,
            "cdc_fluview_who_nrevss": _get_cdc_fluview_who_nrevss_gen_func,
            "project_tycho": _get_project_tycho_gen_func,
        }[dataset](data_path)

        hf_dataset = datasets.Dataset.from_generator(
            gen_func,
            features=features,
            cache_dir=env.HF_CACHE_PATH,
        )
        hf_dataset.info.dataset_name = dataset
        hf_dataset.save_to_disk(self.storage_path / dataset)
