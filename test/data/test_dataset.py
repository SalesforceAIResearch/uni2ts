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

from itertools import cycle

import numpy as np
import pytest
from datasets import load_from_disk

from uni2ts.common.typing import UnivarTimeSeries
from uni2ts.data.dataset import (
    EvalDataset,
    MultiSampleTimeSeriesDataset,
    SampleTimeSeriesType,
    TimeSeriesDataset,
)
from uni2ts.data.indexer import HuggingFaceDatasetIndexer
from uni2ts.transform import Identity


@pytest.mark.parametrize(
    "hf_dataset_path",
    [
        (
            3,
            2,
            1,
            (10, 10),
        ),  # num_examples, target_dim, past_feat_dynamic_real_dim, length
        (3, 2, 5, (10, 20)),
    ],
    indirect=True,
)
def test_global_dataset(hf_dataset_path):
    (
        dataset_path,
        num_examples,
        target_dim,
        past_feat_dynamic_real_dim,
        length,
    ) = hf_dataset_path
    hf_dataset_indexer = HuggingFaceDatasetIndexer(load_from_disk(dataset_path))
    dataset = TimeSeriesDataset(hf_dataset_indexer, transform=Identity())

    assert len(dataset) == num_examples
    for sample_a, sample_b in zip(dataset, cycle(hf_dataset_indexer)):
        for a, b in zip(sample_a.values(), sample_b.values()):
            assert np.all(a == b)


@pytest.mark.parametrize(
    "hf_dataset_path",
    [
        (
            100,
            1,
            None,
            (10, 10),
        ),  # num_examples, target_dim, past_feat_dynamic_real_dim, length
        (100, 2, None, (10, 10)),
        (100, 2, 5, (10, 10)),
    ],
    indirect=True,
)
def test_multi_dataset(hf_dataset_path):
    (
        dataset_path,
        num_examples,
        target_dim,
        past_feat_dynamic_real_dim,
        length,
    ) = hf_dataset_path
    hf_dataset_indexer = HuggingFaceDatasetIndexer(load_from_disk(dataset_path))
    np.random.seed(0)
    dataset = MultiSampleTimeSeriesDataset(
        hf_dataset_indexer,
        Identity(),
        100,
        ("target", "past_feat_dynamic_real"),
    )

    assert len(dataset) == len(hf_dataset_indexer) == num_examples

    sample = dataset[0]

    np.random.seed(0)
    n_series = dataset.sampler(num_examples)
    ts_length = np.random.default_rng(0).integers(*length, endpoint=True)

    target = sample["target"]
    assert isinstance(target, list)
    assert all(isinstance(ts, UnivarTimeSeries) for ts in target)
    assert len(target) == target_dim * n_series
    assert all(len(ts) == ts_length for ts in target)

    if past_feat_dynamic_real_dim is not None:
        feat = sample["past_feat_dynamic_real"]
        assert isinstance(feat, list)
        assert all(isinstance(ts, UnivarTimeSeries) for ts in feat)
        assert len(feat) == past_feat_dynamic_real_dim * n_series
        assert all(len(ts) == ts_length for ts in feat)


@pytest.mark.parametrize(
    "hf_dataset_path",
    [
        (
            3,
            2,
            1,
            (10, 10),
        ),  # num_examples, target_dim, past_feat_dynamic_real_dim, length
        (3, 2, 5, (10, 20)),
    ],
    indirect=True,
)
@pytest.mark.parametrize("weight", [0.5, 1.0, 1.5, 2, 10.3, 20.0])
@pytest.mark.parametrize(
    "sample_time_series",
    [
        SampleTimeSeriesType.NONE,
        SampleTimeSeriesType.UNIFORM,
        SampleTimeSeriesType.PROPORTIONAL,
    ],
)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_weighted_dataset(
    hf_dataset_path, weight: float, sample_time_series: SampleTimeSeriesType, seed: int
):
    (
        dataset_path,
        num_examples,
        target_dim,
        past_feat_dynamic_real_dim,
        length,
    ) = hf_dataset_path
    hf_dataset_indexer = HuggingFaceDatasetIndexer(load_from_disk(dataset_path))
    dataset = TimeSeriesDataset(
        hf_dataset_indexer,
        transform=Identity(),
        sample_time_series=sample_time_series,
        dataset_weight=weight,
    )

    if sample_time_series == SampleTimeSeriesType.UNIFORM:
        assert (dataset.probabilities == 1 / num_examples).all()
    elif sample_time_series == SampleTimeSeriesType.PROPORTIONAL:
        lengths = np.asarray(
            [sample["target"].shape[-1] for sample in hf_dataset_indexer]
        )
        probs = lengths / lengths.sum()
        assert np.allclose(dataset.probabilities, probs)

    for i in range(2):
        assert len(dataset) == int(np.ceil(num_examples * weight))
        ds_iter = iter(dataset)
        idx_iter = iter(cycle(hf_dataset_indexer))
        for j in range(len(dataset)):
            np.random.seed(seed + i * 2 + j)
            sample_a = next(ds_iter)

            if sample_time_series != SampleTimeSeriesType.NONE:
                np.random.seed(seed + i * 2 + j)
                sample_b = hf_dataset_indexer[
                    np.random.choice(len(hf_dataset_indexer), p=dataset.probabilities)
                ]
            else:
                sample_b = next(idx_iter)

            for a, b in zip(sample_a.values(), sample_b.values()):
                assert np.all(a == b)


@pytest.mark.parametrize(
    "hf_dataset_path",
    [
        (
            3,
            2,
            1,
            (1000, 1000),
        ),  # num_examples, target_dim, past_feat_dynamic_real_dim, length
        (3, 2, 5, (1000, 2000)),
    ],
    indirect=True,
)
@pytest.mark.parametrize(
    "windows",
    [10, 10, 91],
)
def test_eval_dataset(
    hf_dataset_path,
    windows: int,
):
    (
        dataset_path,
        num_examples,
        target_dim,
        past_feat_dynamic_real_dim,
        length,
    ) = hf_dataset_path
    hf_dataset_indexer = HuggingFaceDatasetIndexer(load_from_disk(dataset_path))
    dataset = EvalDataset(
        windows,
        hf_dataset_indexer,
        transform=Identity(),
    )

    assert len(dataset) == num_examples * windows

    for idx, sample in enumerate(dataset):
        window, id = divmod(idx, num_examples)
        assert sample["window"] == window
        assert sample["item_id"] == f"item_{id}"
