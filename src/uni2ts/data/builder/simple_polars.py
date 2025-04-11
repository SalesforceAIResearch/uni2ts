import argparse
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Callable, Generator, Optional, List, Union

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
from functools import partial

from ._base import DatasetBuilder
from concurrent.futures import ProcessPoolExecutor, as_completed

ID_COLUMN = "objectid"
TIMESTEMP_COLUMN = "mjd_r"
COLUMNS = ["mag_r", "magerr_r"]

HF_ID_COLUMN = "item_id"
HF_FREQ_COLUMN = "freq"
HF_START_COLUMN = "start"
HF_NON_NULL_COUNTER = "non_null_columns"
HF_TIMESERIES_COLUMN = "target"
    

def mjd_to_unix_timestemp(timestemp):
    return (timestemp - 40587) * 86400 * 1e3

def _transform_polars(lf: pl.LazyFrame, offset: float, end: float, freq: str) -> pl.LazyFrame:
    rows = lf.select(pl.len()).collect().item()
    offset_int = int(rows * offset)
    length = int(rows * end) - offset_int

    lf = lf.slice(offset_int, length)
    
    select_id_expr = pl.col(ID_COLUMN).alias(HF_ID_COLUMN)
    first_timestemp_exprs = mjd_to_unix_timestemp(pl.col(TIMESTEMP_COLUMN).list.first()).cast(pl.Datetime("ms")).alias(HF_START_COLUMN)

    non_null_columns = [pl.when(pl.col(col_name).is_not_null()).then(pl.col(col_name)) for col_name in COLUMNS]
    concatenation_exprs = pl.concat_list(non_null_columns).alias(HF_TIMESERIES_COLUMN)

    non_null_flags = [pl.col(col_name).is_not_null().cast(pl.UInt8) for col_name in COLUMNS]
    not_null_column_count_expr = pl.fold(
        acc=pl.lit(0), function=lambda acc, x: acc + x, exprs=non_null_flags
    ).alias(HF_NON_NULL_COUNTER)
    
    lf = lf.select(select_id_expr, first_timestemp_exprs, concatenation_exprs, not_null_column_count_expr)
    lf = lf.with_columns(
        pl.lit(freq).alias(HF_FREQ_COLUMN)
    )
    
    features = Features(
        {
            HF_ID_COLUMN: Value("string"),
            HF_START_COLUMN: Value("timestamp[ms]"),
            HF_FREQ_COLUMN: Value("string"),
            HF_NON_NULL_COUNTER: Value("uint8"),
            HF_TIMESERIES_COLUMN: Sequence(Value("float32")),
        }
    )
    
    return lf, features

def _load_and_transform_polars(path, offset: float, end: float, freq: str):
    lf = pl.scan_parquet(path)
    lf, features = _transform_polars(lf, offset, end, freq)
    df = lf.collect()
    
    return datasets.Dataset.from_polars(df, features=features)
    

def _create_hf_dataset_from_polars(files: List[str], offset: float = 0.0, end: float = 1.0, freq: str = "H", max_workers: Optional[int] = None):
    if offset > 1.0 or offset < 0.0:
        raise ValueError(f"Offset should be a positive integer or a float between 0 and 1.")

    if end > 1.0 or end < 0.0:
        raise ValueError(f"End should be a positive integer or a float between 0 and 1.")
    
    polars_transform_fun = partial(_load_and_transform_polars, offset = offset, end = end, freq = freq)
    
    if not max_workers:
        max_workers = max(1, os.cpu_count() - 1)
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        dataset = list(executor.map(polars_transform_fun, files))
        
    return datasets.concatenate_datasets(dataset)

def _select_parquet_files(folder_path):
    return [
            path_object
            for path_object in folder_path.iterdir()
            if path_object.is_file() and path_object.suffix == ".parquet"
        ]


def _transform_to_uni2ts_format(batch):
    timeseries = batch[HF_TIMESERIES_COLUMN]
    non_null_counter = batch.pop(HF_NON_NULL_COUNTER)
    
    transformed_timeseries = []
    
    for data, non_null_count in zip(timeseries, non_null_counter):
        transformed_data = np.array(data).reshape((non_null_count, -1))
        transformed_timeseries.append(transformed_data)
        
    batch[HF_TIMESERIES_COLUMN] = transformed_timeseries
    
    return batch

@dataclass
class SimplePolarsDatasetBuilder(DatasetBuilder):
    dataset: str
    weight: float = 1.0
    sample_time_series: Optional[SampleTimeSeriesType] = SampleTimeSeriesType.NONE
    storage_path: Path = env.CUSTOM_DATA_PATH

    def __post_init__(self):
        self.storage_path = Path(self.storage_path)

    def build_dataset(self, folder_path: Path, offset: float, end: float, freq: str, num_workers=None):
        polars_files = _select_parquet_files(folder_path)
        hf_dataset = _create_hf_dataset_from_polars(polars_files, offset, end, freq)
        
        hf_dataset.info.dataset_name = self.dataset
        hf_dataset.save_to_disk(str(self.storage_path / self.dataset))

    def load_dataset(self, transform_map: dict[str, Callable[..., Transformation]]) -> Dataset:
        hf_dataset = datasets.load_from_disk(str(self.storage_path / self.dataset))
        hf_dataset.set_transform(_transform_to_uni2ts_format)
        
        return TimeSeriesDataset(
            HuggingFaceDatasetIndexer(hf_dataset),
            transform=transform_map[self.dataset](),
            dataset_weight=self.weight,
            sample_time_series=self.sample_time_series,
        )


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

    def build_dataset(self, folder_path: Path, offset: float, end: float, freq: str, num_workers=None):
        polars_files = _select_parquet_files(folder_path)
        hf_dataset = _create_hf_dataset_from_polars(polars_files, offset, end, freq)
        
        hf_dataset.info.dataset_name = self.dataset
        hf_dataset.save_to_disk(str(self.storage_path / self.dataset))

    def load_dataset(self, transform_map: dict[str, Callable[..., Transformation]]) -> Dataset:
        hf_dataset = datasets.load_from_disk(str(self.storage_path / self.dataset))
        hf_dataset.set_transform(_transform_to_uni2ts_format)
        
        return EvalDataset(
            self.windows,
            HuggingFaceDatasetIndexer(
                
            ),
            transform=transform_map[self.dataset](
                offset=self.offset,
                distance=self.distance,
                prediction_length=self.prediction_length,
                context_length=self.context_length,
                patch_size=self.patch_size,
            ),
        )


def build_datasets(args):
    dataset_builder = SimplePolarsDatasetBuilder(dataset=args.dataset_name)
    dataset_builder.build_dataset(folder_path=Path(args.folder_path), offset=0.0, end=args.split_ratio, freq=args.freq)

    if args.freq == 1.0:
        return

    eval_dataset_builder = SimpleEvalDatasetBuilder(
        dataset=f"{args.dataset_name}_eval",
        offset=None,
        windows=None,
        distance=None,
        prediction_length=None,
        context_length=None,
        patch_size=None,
    )
    eval_dataset_builder.build_dataset(
        folder_path=Path(args.folder_path), offset=args.split_ratio, end=1.0, freq=args.freq
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_name", type=str)
    parser.add_argument("folder_path", type=str)
    # parser.add_argument("--timestemp_column", type=str)
    # parser.add_argument("--columns", type=str, nargs="+")
    parser.add_argument("--split_ratio", type=float, default=0.8)
    # Define the `freq` argument with a default value. Use this value as 'freq' if 'freq' is None.
    parser.add_argument(
        "--freq",
        default="H",  # Set the default value
        help="The user specified frequency",
    )

    args = parser.parse_args()
    build_datasets(args)
