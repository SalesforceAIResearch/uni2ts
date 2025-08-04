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

from enum import Enum
from typing import Any

import numpy as np
from torch.utils.data import Dataset

from uni2ts.common.sampler import Sampler, get_sampler
from uni2ts.common.typing import (
    BatchedData,
    BatchedDateTime,
    BatchedString,
    Data,
    FlattenedData,
    MultivarTimeSeries,
    UnivarTimeSeries,
)
from uni2ts.data.indexer import Indexer
from uni2ts.transform import Transformation


class SampleTimeSeriesType(Enum):
    """
    An enumeration that defines how to sample time series from a dataset.

    Attributes:
        NONE: Do not sample; return the time series at the current index.
        UNIFORM: Sample each time series with equal probability.
        PROPORTIONAL: Sample each time series with a probability proportional to its length.
    """

    NONE = "none"
    UNIFORM = "uniform"
    PROPORTIONAL = "proportional"


class TimeSeriesDataset(Dataset):
    """
    A PyTorch Dataset for handling time series data. It wraps an Indexer and applies a
    transformation to the data. It also supports different sampling strategies for
    retrieving time series.

    Args:
        indexer (Indexer[dict[str, Any]]): The underlying Indexer object that provides access to the data.
        transform (Transformation): A transformation to apply to the time series data.
        sample_time_series (SampleTimeSeriesType, optional): The sampling strategy to use.
            Defaults to SampleTimeSeriesType.NONE.
        dataset_weight (float, optional): A multiplicative factor to apply to the dataset size.
            Defaults to 1.0.
    """
    def __init__(
        self,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        dataset_weight: float = 1.0,
    ):
        self.indexer = indexer
        self.transform = transform
        self.sample_time_series = sample_time_series
        self.dataset_weight = dataset_weight

        if sample_time_series == SampleTimeSeriesType.NONE:
            self.probabilities = None
        elif sample_time_series == SampleTimeSeriesType.UNIFORM:
            self.probabilities = indexer.get_uniform_probabilities()
        elif sample_time_series == SampleTimeSeriesType.PROPORTIONAL:
            self.probabilities = indexer.get_proportional_probabilities()
        else:
            raise ValueError(f"Unknown sample type {sample_time_series}")

    def __getitem__(self, idx: int) -> dict[str, FlattenedData]:
        """
        Retrieves a time series from the dataset, applies a transformation, and returns it.
        If a sampling strategy is specified, `idx` is ignored and a time series is sampled
        according to the strategy.

        Args:
            idx (int): The index of the time series to retrieve.

        Returns:
            dict[str, FlattenedData]: The transformed time series data.
        """
        if idx < 0 or idx >= len(self):
            raise IndexError(
                f"Index {idx} out of range for dataset of length {len(self)}"
            )

        if self.sample_time_series != SampleTimeSeriesType.NONE:
            idx = np.random.choice(len(self.probabilities), p=self.probabilities)

        return self.transform(self._flatten_data(self._get_data(idx)))

    @property
    def num_ts(self) -> int:
        """
        Returns the number of time series in the dataset.
        """
        return len(self.indexer)

    def __len__(self) -> int:
        """
        Returns the length of the dataset, which is the number of time series
        multiplied by the dataset_weight.
        """
        return int(np.ceil(self.num_ts * self.dataset_weight))

    def _get_data(self, idx: int) -> dict[str, Data | BatchedData]:
        """
        Retrieves a time series from the underlying Indexer object.
        """
        return self.indexer[idx % self.num_ts]

    @staticmethod
    def _flatten_data(data: dict[str, Data]) -> dict[str, FlattenedData]:
        """
        Converts multivariate time series data into a list of univariate time series.
        """
        return {
            k: (
                [v]
                if isinstance(v, UnivarTimeSeries)
                else list(v) if isinstance(v, MultivarTimeSeries) else v
            )
            for k, v in data.items()
        }


class MultiSampleTimeSeriesDataset(TimeSeriesDataset):
    """
    A dataset that samples multiple time series and stacks them into a single sample.
    This is useful for creating models that can process multiple time series at once.
    The underlying dataset should have aligned time series (i.e., same start and end dates).

    Args:
        indexer (Indexer[dict[str, Any]]): The underlying Indexer object.
        transform (Transformation): A transformation to apply to the time series data.
        max_ts (int): The maximum number of time series that can be stacked together.
        combine_fields (tuple[str, ...]): A tuple of field names that should be stacked.
        sample_time_series (SampleTimeSeriesType, optional): The sampling strategy to use.
            Defaults to SampleTimeSeriesType.NONE.
        dataset_weight (float, optional): A multiplicative factor to apply to the dataset size.
            Defaults to 1.0.
        sampler (Sampler, optional): A sampler function to determine how many time series to sample.
            Defaults to a beta-binomial sampler.
    """

    def __init__(
        self,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
        max_ts: int,
        combine_fields: tuple[str, ...],
        sample_time_series: SampleTimeSeriesType = SampleTimeSeriesType.NONE,
        dataset_weight: float = 1.0,
        sampler: Sampler = get_sampler("beta_binomial", a=2, b=5),
    ):
        super().__init__(indexer, transform, sample_time_series, dataset_weight)
        self.max_ts = max_ts
        self.combine_fields = combine_fields
        self.sampler = sampler

    def _get_data(self, idx: int) -> dict[str, BatchedData]:
        """
        Retrieves multiple time series from the indexer and combines them.
        """
        n_series = self.sampler(min(self.num_ts, self.max_ts))
        choices = np.concatenate([np.arange(idx), np.arange(idx + 1, self.num_ts)])
        others = np.random.choice(choices, n_series - 1, replace=False)
        samples = self.indexer[np.concatenate([[idx], others])]
        return samples

    def _flatten_data(
        self, samples: dict[str, BatchedData]
    ) -> dict[str, FlattenedData]:
        """
        Flattens the combined time series data. For fields specified in `combine_fields`,
        it combines the list of multivariate time series into a single list of univariate
        time series. For other fields, it takes the first element.
        """
        for field in samples.keys():
            if field in self.combine_fields:
                item = samples[field]
                if isinstance(item, list) and isinstance(item[0], MultivarTimeSeries):
                    samples[field] = [
                        univar for sample in samples[field] for univar in sample
                    ]
            elif isinstance(samples[field], BatchedDateTime):
                samples[field] = np.asarray(samples[field][0])
            elif isinstance(samples[field], BatchedString):
                samples[field] = samples[field][0]
            else:
                raise AssertionError(
                    f"Field {field} not accounted for in {self.indexer} MultiSampleTimeSeriesDataset"
                )
        return samples


class EvalDataset(TimeSeriesDataset):
    """
    A dataset class specifically for evaluation. It is designed to be used with
    evaluation-specific transformations. It creates multiple evaluation windows
    for each time series in the dataset.

    Args:
        windows (int): The number of evaluation windows to create for each time series.
        indexer (Indexer[dict[str, Any]]): The underlying Indexer object.
        transform (Transformation): A transformation to apply to the time series data.
    """

    def __init__(
        self,
        windows: int,
        indexer: Indexer[dict[str, Any]],
        transform: Transformation,
    ):
        super().__init__(
            indexer,
            transform,
            SampleTimeSeriesType.NONE,
            dataset_weight=windows,
        )

    def _get_data(self, idx: int) -> dict[str, Data]:
        """
        Retrieves a time series and adds a "window" key to it, which can be used
        by evaluation transformations.
        """
        window, idx = divmod(idx, self.num_ts)
        item = self.indexer[idx]
        item["window"] = window
        return item
