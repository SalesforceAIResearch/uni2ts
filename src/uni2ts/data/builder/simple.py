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

import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Generator, Optional

import datasets
import numpy as np
import pandas as pd
from datasets import Features, Sequence, Value
from torch.utils.data import Dataset

from uni2ts.common.env import env
from uni2ts.common.typing import GenFunc
from uni2ts.data.dataset import EvalDataset, SampleTimeSeriesType, TimeSeriesDataset
from uni2ts.data.indexer import HuggingFaceDatasetIndexer
from uni2ts.transform import Identity, Transformation

# from ._base import DatasetBuilder
from uni2ts.data.builder._base import DatasetBuilder

def _from_long_dataframe(
    df: pd.DataFrame,
) -> tuple[GenFunc, Features]:
    items = df.item_id.unique()

    def example_gen_func() -> Generator[dict[str, Any], None, None]:
        for item_id in items:
            item_df = df.query(f'item_id == "{item_id}"').drop("item_id", axis=1)
            yield {
                "target": item_df.to_numpy(),
                "start": item_df.index[0],
                "freq": pd.infer_freq(item_df.index),
                "item_id": item_id,
            }

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            freq=Value("string"),
            target=Sequence(Value("float32")),
        )
    )

    return example_gen_func, features


def _from_wide_dataframe(
    df: pd.DataFrame,
) -> tuple[GenFunc, Features]:
    def example_gen_func() -> Generator[dict[str, Any], None, None]:
        for i in range(len(df.columns)):
            yield {
                "target": df.iloc[:, i].to_numpy(),
                "start": df.index[0],
                "freq": pd.infer_freq(df.index),
                "item_id": f"item_{i}",
            }

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            freq=Value("string"),
            target=Sequence(Value("float32")),
        )
    )

    return example_gen_func, features


def _from_wide_dataframe_multivariate(
    df: pd.DataFrame,
) -> tuple[GenFunc, Features]:
    def example_gen_func() -> Generator[dict[str, Any], None, None]:
        yield {
            "target": df.to_numpy().T,
            "start": df.index[0],
            "freq": pd.infer_freq(df.index),
            "item_id": "item_0",
        }

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("timestamp[s]"),
            freq=Value("string"),
            target=Sequence(Sequence(Value("float32")), length=len(df.columns)),
        )
    )

    return example_gen_func, features


@dataclass
class SimpleDatasetBuilder(DatasetBuilder):
    dataset: str
    weight: float = 1.0
    sample_time_series: Optional[SampleTimeSeriesType] = SampleTimeSeriesType.NONE
    storage_path: Path = env.CUSTOM_DATA_PATH

    def __post_init__(self):
        self.storage_path = Path(self.storage_path)
        self.mean = None
        self.std = None

    def build_dataset(
        self,
        file: Path,
        dataset_type: str,
        offset: int = None,
        date_offset: pd.Timestamp = None,
        normalize: bool = False
    ):
        """
        Build a Hugging Face dataset based on the csv file of the entire TS dataset.

        Only use the range before offset/date_offset if they are specified.

        If dataset_type == 'long':
            data must have an 'item_id' column. Treat each item's record as a data_entry.
        If dataset_type == 'wide':
            hf_dataset treats the whole record of each individual channel as a data_entry.
            Number of data_entry is the same as the number of channels/variates.
        If dataset_type == 'wide_multivariate':
            All the channels/variates will be treated as a whole. 'target' is a MTS.
            Only 1 entry in hf_dataset.
        """

        assert offset is None or date_offset is None, (
            "One or neither offset and date_offset must be specified, but not both. "
            f"Got offset: {offset}, date_offset: {date_offset}"
        )

        df = pd.read_csv(file, index_col=0, parse_dates=True)  # Use date as index

        if offset is not None:
            df = df.iloc[:offset]

        if date_offset is not None:
            df = df[df.index <= date_offset]

        if normalize:
            df = self.scale(df, 0, len(df.index))

        if dataset_type == "long":
            example_gen_func, features = _from_long_dataframe(df)
        elif dataset_type == "wide":
            example_gen_func, features = _from_wide_dataframe(df)
        elif dataset_type == "wide_multivariate":
            example_gen_func, features = _from_wide_dataframe_multivariate(df)
        else:
            raise ValueError(
                f"Unrecognized dataset_type, {dataset_type}."
                " Valid options are 'long', 'wide', and 'wide_multivariate'."
            )

        hf_dataset = datasets.Dataset.from_generator(
            example_gen_func, features=features
        )
        hf_dataset.info.dataset_name = self.dataset
        hf_dataset.save_to_disk(self.storage_path / self.dataset)

    def load_dataset(self, transform: Transformation = Identity(), *args) -> Dataset:
        """
        Apply uni2ts transformation to dataset
        """

        return TimeSeriesDataset(
            HuggingFaceDatasetIndexer(
                datasets.load_from_disk(
                    str(self.storage_path / self.dataset),
                )
            ),
            transform=transform,
            dataset_weight=self.weight,
            sample_time_series=self.sample_time_series,
        )

    def scale(self, data, start, end):
        train = data[start:end]
        self.mean = train.mean(axis=0)
        self.std = train.std(axis=0)
        return (data - self.mean) / self.std


@dataclass
class SimpleEvalDatasetBuilder(DatasetBuilder):
    dataset: str
    offset: Optional[int]
    windows: Optional[int]
    distance: Optional[int]
    prediction_length: Optional[int]
    context_length: Optional[int]
    patch_size: Optional[int]
    storage_path: Path = env.CUSTOM_DATA_PATH

    def __post_init__(self):
        self.storage_path = Path(self.storage_path)

    def build_dataset(self, file: Path, dataset_type: str, mean: pd.Series = None, std: pd.Series = None):
        """
        Same as above. But no offset. Save the entire dataset.
        """
        df = pd.read_csv(file, index_col=0, parse_dates=True)

        if mean is not None and std is not None:
            df = (df - mean) / std

        if dataset_type == "long":
            example_gen_func, features = _from_long_dataframe(df)
        elif dataset_type == "wide":
            example_gen_func, features = _from_wide_dataframe(df)
        elif dataset_type == "wide_multivariate":
            example_gen_func, features = _from_wide_dataframe_multivariate(df)
        else:
            raise ValueError(
                f"Unrecognized dataset_type, {dataset_type}."
                " Valid options are 'long', 'wide', and 'wide_multivariate'."
            )

        hf_dataset = datasets.Dataset.from_generator(
            example_gen_func, features=features
        )
        hf_dataset.info.dataset_name = self.dataset
        hf_dataset.save_to_disk(self.storage_path / self.dataset)

    def load_dataset(
        self, get_transform: Callable[[int, int, int, int, int], Transformation]
    ) -> Dataset:
        """
        EvalDataset, which specifies windows, distance, prediction_length, context_length, patch_size.
        """
        return EvalDataset(
            self.windows,
            HuggingFaceDatasetIndexer(
                datasets.load_from_disk(
                    str(self.storage_path / self.dataset),
                )
            ),
            transform=get_transform(
                self.offset,
                self.distance,
                self.prediction_length,
                self.context_length,
                self.patch_size,
            ),
        )


def generate_eval_builders(
    dataset: str,
    offset: int,
    eval_length: int,
    prediction_lengths: list[int],
    context_lengths: list[int],
    patch_sizes: list[int],
    storage_path: Path = env.CUSTOM_DATA_PATH,
) -> list[SimpleEvalDatasetBuilder]:
    return [
        SimpleEvalDatasetBuilder(
            dataset=dataset,
            offset=offset,
            windows=eval_length // pred,
            distance=pred,
            prediction_length=pred,
            context_length=ctx,
            patch_size=psz,
            storage_path=storage_path,
        )
        for pred, ctx, psz in product(prediction_lengths, context_lengths, patch_sizes)
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("file_path", type=str)
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["wide", "long", "wide_multivariate"],
        default="wide",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--date_offset",
        type=str,
        default=None,
    )
    parser.add_argument("--normalize", action="store_true")
    args = parser.parse_args()

    # Create training dataset
    # If offset/date_offset is not provided, the whole data will be used for training
    # Otherwise, only the part before offset is used for training.
    train_dataset_builder = SimpleDatasetBuilder(dataset=args.dataset_name)
    train_dataset_builder.build_dataset(
        file=Path(args.file_path),
        dataset_type=args.dataset_type,
        offset=args.offset,
        date_offset=pd.Timestamp(args.date_offset) if args.date_offset else None,
        normalize=args.normalize  # To align with LSF. Default is False
    )

    # Create a validation dataset if offset/data_offset is provided.
    # Eval dataset include the whole data.
    # 'offsets' and 'windows' need to be specified in yaml.
    if args.offset is not None or args.date_offset is not None:
        SimpleEvalDatasetBuilder(
            f"{args.dataset_name}_eval",
            offset=None,
            windows=None,
            distance=None,
            prediction_length=None,
            context_length=None,
            patch_size=None,
        ).build_dataset(file=Path(args.file_path),
                        dataset_type=args.dataset_type,
                        mean=train_dataset_builder.mean,
                        std=train_dataset_builder.std)
