import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Generator, Optional, List

import polars as pl

import datasets
from datasets import Features, Sequence, Value
from torch.utils.data import Dataset

from uni2ts.common.env import env
from uni2ts.common.typing import GenFunc
from uni2ts.data.dataset import EvalDataset, SampleTimeSeriesType, TimeSeriesDataset
from uni2ts.data.indexer import HuggingFaceDatasetIndexer
from uni2ts.transform import Transformation
import numpy as np
import os

from ._base import DatasetBuilder


def _from_polars(
    timestemp_column: str,
    columns: List[str],
    freq: str = "H",
) -> tuple[GenFunc, Features]:
    START_COLUMN = "start"
    TIMESERIES_COLUMN = "time_series"
    NON_NULL_COUNTER = "non_null_columns"

    def generator_function(shards: List[str]) -> Generator[dict[str, Any], None, None]:
        # Do not move Polars expressions below outside of the generator func as those are not serializable with dill
        start_exprs = pl.col(timestemp_column).list.first().alias(START_COLUMN)

        non_null_columns = [pl.when(pl.col(col_name).is_not_null()).then(pl.col(col_name)) for col_name in columns]
        concatenation_exprs = pl.concat_list(non_null_columns).alias(TIMESERIES_COLUMN)

        non_null_flags = [pl.col(col_name).is_not_null().cast(pl.UInt8) for col_name in columns]
        not_null_column_count_expr = pl.fold(
            acc=pl.lit(0), function=lambda acc, x: acc + x, exprs=non_null_flags
        ).alias(NON_NULL_COUNTER)

        for shard in shards:
            lf = pl.scan_parquet(shard)
            lf = lf.with_columns(start_exprs, concatenation_exprs, not_null_column_count_expr).collect()

            for row in lf.iter_rows(named=True):
                data = np.array(row[TIMESERIES_COLUMN]).reshape((row[NON_NULL_COUNTER], -1))

                yield {"target": data, "start": row[START_COLUMN], "freq": freq, "item_id": f"{shard}_{row}"}

    features = Features(
        dict(
            item_id=Value("string"),
            start=Value("float32"),
            freq=Value("string"),
            target=Sequence(Sequence(Value("float32"))),
        )
    )

    return generator_function, features


@dataclass
class SimplePolarsDatasetBuilder(DatasetBuilder):
    dataset: str
    timestemp_column: str
    columns: list[str]
    weight: float = 1.0
    sample_time_series: Optional[SampleTimeSeriesType] = SampleTimeSeriesType.NONE
    storage_path: Path = env.CUSTOM_DATA_PATH

    def __post_init__(self):
        self.storage_path = Path(self.storage_path)

    def build_dataset(self, folder_path: Path, freq: str = "H", num_workers=None):
        generator_func, features = _from_polars(self.timestemp_column, self.columns, freq)
        polars_files = [path_object for path_object in folder_path.iterdir() if path_object.is_file()]

        hf_dataset = datasets.Dataset.from_generator(
            generator_func,
            features=features,
            gen_kwargs={"shards": polars_files},
            num_proc=num_workers if num_workers else os.cpu_count(),
        )

        hf_dataset.info.dataset_name = self.dataset
        hf_dataset.save_to_disk(self.storage_path / self.dataset)

    def load_dataset(self, transform_map: dict[str, Callable[..., Transformation]]) -> Dataset:
        return TimeSeriesDataset(
            HuggingFaceDatasetIndexer(
                datasets.load_from_disk(
                    str(self.storage_path / self.dataset),
                )
            ),
            transform=transform_map[self.dataset](),
            dataset_weight=self.weight,
            sample_time_series=self.sample_time_series,
        )


@dataclass
class SimplePolarsEvalDatasetBuilder(DatasetBuilder):
    dataset: str
    timestemp_column: str
    columns: List[str]
    offset: Optional[int]
    windows: Optional[int]
    distance: Optional[int]
    prediction_length: Optional[int]
    context_length: Optional[int]
    patch_size: Optional[int]
    storage_path: Path = env.CUSTOM_DATA_PATH

    def __post_init__(self):
        self.storage_path = Path(self.storage_path)

    def build_dataset(self, folder_path: Path, freq: str = "H", num_workers=None):
        generator_func, features = _from_polars(self.timestemp_column, self.columns, freq)
        polars_files = [path_object for path_object in folder_path.iterdir() if path_object.is_file()]

        hf_dataset = datasets.Dataset.from_generator(
            generator_func,
            features=features,
            gen_kwargs={"shards": polars_files},
            num_proc=num_workers if num_workers else os.cpu_count(),
        )

        hf_dataset.info.dataset_name = self.dataset
        hf_dataset.save_to_disk(self.storage_path / self.dataset)

    def load_dataset(self, transform_map: dict[str, Callable[..., Transformation]]) -> Dataset:
        return EvalDataset(
            self.windows,
            HuggingFaceDatasetIndexer(
                datasets.load_from_disk(
                    str(self.storage_path / self.dataset),
                )
            ),
            transform=transform_map[self.dataset](
                offset=self.offset,
                distance=self.distance,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                patch_size=self.patch_size,
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
) -> list[SimplePolarsEvalDatasetBuilder]:
    return [
        SimplePolarsEvalDatasetBuilder(
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
    parser.add_argument("folder_path", type=str)
    parser.add_argument("--timestemp_column", type=str)
    parser.add_argument("--columns", type=str, nargs="+")
    # Define the `freq` argument with a default value. Use this value as 'freq' if 'freq' is None.
    parser.add_argument(
        "--freq",
        default="H",  # Set the default value
        help="The user specified frequency",
    )

    args = parser.parse_args()

    dataset_builder = SimplePolarsDatasetBuilder(
        dataset=args.dataset_name, timestemp_column=args.timestemp_column, columns=args.columns
    )

    dataset_builder.build_dataset(folder_path=Path(args.folder_path), freq=args.freq)
