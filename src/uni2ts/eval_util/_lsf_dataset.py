import os

import numpy as np
import pandas as pd

from uni2ts.common.env import env


class LSFDataset:
    def __init__(
        self,
        dataset_name: str,
        mode: str = "S",
        split: str = "test",
    ):
        self.dataset_name = dataset_name
        self.mode = mode
        self.split = split

        if dataset_name in ["ETTh1", "ETTh2"]:
            self._load_etth()
        elif dataset_name in ["ETTm1", "ETTm2"]:
            self._load_ettm()
        elif dataset_name == "METR_LA":
            self._load_metr_la()
        elif dataset_name == "solar":
            self._load_solar()
        elif dataset_name == "walmart":
            self._load_walmart()
        elif dataset_name == "electricity":
            self._load_custom("electricity/electricity.csv", "h")
        elif dataset_name == "weather":
            self._load_custom("weather/weather.csv", "10T")
        else:
            raise ValueError(f"Unknown dataset name: {dataset_name}")

        if mode == "S":
            self.target_dim = 1
            self.past_feat_dynamic_real_dim = 0
        elif mode == "M":
            self.target_dim = self.data.shape[-1]
            self.past_feat_dynamic_real_dim = 0
        elif mode == "MS":
            self.target_dim = 1
            self.past_feat_dynamic_real_dim = self.data.shape[-1] - 1
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def __iter__(self):
        if self.mode == "S":
            for i in range(self.data.shape[-1]):
                yield {
                    "target": self.data[:, i],
                    "start": self.start,
                }
        elif self.mode == "M":
            yield {
                "target": self.data.transpose(1, 0),
                "start": self.start,
            }
        elif self.mode == "MS":
            for i in range(self.data.shape[-1]):
                yield {
                    "target": self.data[:, i],
                    "past_feat_dynamic_real": np.concatenate(
                        [self.data[:, :i], self.data[:, i + 1 :]], axis=1
                    ).transpose(1, 0),
                    "start": self.start,
                }

    def scale(self, data, start, end):
        train = data[start:end]
        mean = train.mean(axis=0)
        std = train.std(axis=0)
        return (data - mean) / std

    def _load_etth(self):
        df = pd.read_csv(
            os.path.join(env.LSF_PATH, f"ETT-small/{self.dataset_name}.csv")
        )

        train_length = 8640
        val_length = 2880
        test_length = 2880
        data = self.scale(df[df.columns[1:]], 0, train_length).to_numpy()
        if self.split == "train":
            self.data = data[:train_length]
            self.length = train_length
        elif self.split == "val":
            self.data = data[: train_length + val_length]
            self.length = val_length
        elif self.split == "test":
            self.data = data[: train_length + val_length + test_length]
            self.length = test_length
        self.start = pd.to_datetime(df[["date"]].iloc[0].item())
        self.freq = "h"

    def _load_ettm(self):
        df = pd.read_csv(
            os.path.join(env.LSF_PATH, f"ETT-small/{self.dataset_name}.csv")
        )

        train_length = 34560
        val_length = 11520
        test_length = 11520
        data = self.scale(df[df.columns[1:]], 0, train_length).to_numpy()
        if self.split == "train":
            self.data = data[:train_length]
            self.length = train_length
        elif self.split == "val":
            self.data = data[: train_length + val_length]
            self.length = val_length
        elif self.split == "test":
            self.data = data[: train_length + val_length + test_length]
            self.length = test_length
        self.start = pd.to_datetime(df[["date"]].iloc[0].item())
        self.freq = "15T"

    def _load_solar(self):
        df = pd.read_csv(os.path.join(env.LSF_PATH, "Solar/solar_AL.txt"), header=None)
        data = df.to_numpy().reshape(8760, 6, 137).sum(1)

        train_length = int(len(data) * 0.7)
        val_length = int(len(data) * 0.1)
        test_length = int(len(data) * 0.2)

        data = self.scale(data, 0, train_length).to_numpy()
        if self.split == "train":
            self.data = data[:train_length]
            self.length = train_length
        elif self.split == "val":
            self.data = data[: train_length + val_length]
            self.length = val_length
        elif self.split == "test":
            self.data = data[: train_length + val_length + test_length]
            self.length = test_length
        self.start = pd.to_datetime("2006-01-01")
        self.freq = "h"

    def _load_metr_la(self):
        df = pd.read_csv(os.path.join(env.LSF_PATH, "METR_LA/METR_LA.dyna"))
        data = []
        for id in df.entity_id.unique():
            id_df = df[df.entity_id == id]
            data.append(id_df.traffic_speed.to_numpy())
        data = np.stack(data, 1)

        train_length = int(len(data) * 0.7)
        val_length = int(len(data) * 0.1)
        test_length = int(len(data) * 0.2)

        data = self.scale(data, 0, train_length).to_numpy()
        if self.split == "train":
            self.data = data[:train_length]
            self.length = train_length
        elif self.split == "val":
            self.data = data[: train_length + val_length]
            self.length = val_length
        elif self.split == "test":
            self.data = data[: train_length + val_length + test_length]
            self.length = test_length
        self.start = pd.to_datetime("2012-03-01")
        self.freq = "5T"

    def _load_walmart(self):
        df = pd.read_csv(
            os.path.join(
                env.LSF_PATH, "walmart-recruiting-store-sales-forecasting/train.csv"
            )
        )

        data = []
        for id, row in df[["Store", "Dept"]].drop_duplicates().iterrows():
            row_df = df.query(f"Store == {row.Store} and Dept == {row.Dept}")
            if len(row_df) != 143:
                continue
            data.append(row_df.Weekly_Sales.to_numpy())
        data = np.stack(data, 1)

        train_length = 143 - 28 - 14
        val_length = 14
        test_length = 28
        data = self.scale(data, 0, train_length)
        if self.split == "train":
            self.data = data[:train_length]
            self.length = train_length
        elif self.split == "val":
            self.data = data[: train_length + val_length]
            self.length = val_length
        elif self.split == "test":
            self.data = data[: train_length + val_length + test_length]
            self.length = test_length
        self.start = pd.to_datetime("2010-02-05")
        self.freq = "W"

    def _load_custom(self, data_path: str, freq: str):
        df = pd.read_csv(os.path.join(env.LSF_PATH, data_path))
        cols = list(df.columns)
        cols.remove("OT")
        cols.remove("date")
        df = df[["date"] + cols + ["OT"]]
        data = df[df.columns[1:]]

        train_length = int(len(data) * 0.7)
        val_length = int(len(data) * 0.1)
        test_length = int(len(data) * 0.2)
        data = self.scale(data, 0, train_length).to_numpy()
        if self.split == "train":
            self.data = data[:train_length]
            self.length = train_length
        elif self.split == "val":
            self.data = data[: train_length + val_length]
            self.length = val_length
        elif self.split == "test":
            self.data = data[: train_length + val_length + test_length]
            self.length = test_length
        self.start = pd.to_datetime(df[["date"]].iloc[0].item())
        self.freq = freq
