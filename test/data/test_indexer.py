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

from typing import Optional

import numpy as np
import pytest
from datasets import load_from_disk

from uni2ts.common.typing import (
    BatchedData,
    BatchedDateTime,
    BatchedString,
    Data,
    DateTime,
    MultivarTimeSeries,
    String,
    UnivarTimeSeries,
)
from uni2ts.data.indexer import HuggingFaceDatasetIndexer


def _test_item(
    item: dict[str, Data],
    target_dim: int,
    past_feat_dynamic_real_dim: Optional[int],
    ts_lengths: list[int],
):
    for k in item.keys():
        if k in ("item_id", "freq"):
            assert isinstance(item[k], String)
        elif k in ("start"):
            assert isinstance(item[k], DateTime)
        elif k in ("target", "past_feat_dynamic_real"):
            assert isinstance(item[k], (UnivarTimeSeries, MultivarTimeSeries))
        else:
            raise AssertionError(f"Unexpected key: {k}")

    target_shape = () if target_dim == 1 else (target_dim,)
    assert item["target"].shape == target_shape + (ts_lengths[0],)

    if past_feat_dynamic_real_dim is not None:
        feat_shape = (
            () if past_feat_dynamic_real_dim == 1 else (past_feat_dynamic_real_dim,)
        )
        assert item["past_feat_dynamic_real"].shape == feat_shape + (ts_lengths[0],)


def _test_batch(
    batch: dict[str, BatchedData],
    target_dim: int,
    past_feat_dynamic_real_dim: Optional[int],
    ts_lengths: list[int],
):
    for k in batch.keys():
        if k in ("item_id", "freq"):
            assert isinstance(batch[k], BatchedString)
        elif k in ("start",):
            assert isinstance(batch[k], BatchedDateTime)
        elif k in ("target", "past_feat_dynamic_real"):
            assert isinstance(batch[k], list)
            assert all(
                isinstance(example, (UnivarTimeSeries, MultivarTimeSeries))
                for example in batch[k]
            )
        else:
            raise AssertionError(f"Unexpected key: {k}")

    target_shape = () if target_dim == 1 else (target_dim,)
    for idx, target in enumerate(batch["target"]):
        assert target.shape == target_shape + (ts_lengths[idx],)

    if past_feat_dynamic_real_dim is not None:
        feat_shape = (
            () if past_feat_dynamic_real_dim == 1 else (past_feat_dynamic_real_dim,)
        )
        for idx, past_feat_dynamic_real in enumerate(batch["past_feat_dynamic_real"]):
            assert past_feat_dynamic_real.shape == feat_shape + (ts_lengths[idx],)


@pytest.mark.parametrize(
    "hf_dataset_path",
    [
        (
            3,
            1,
            None,
            (10, 10),
        ),  # num_examples, target_dim, past_feat_dynamic_real_dim, length
        (3, 1, None, (10, 20)),
        (3, 2, 1, (10, 10)),
        (3, 2, 5, (10, 20)),
    ],
    indirect=True,
)
def test_hf_dataset_indexer_int(hf_dataset_path):
    (
        dataset_path,
        num_examples,
        target_dim,
        past_feat_dynamic_real_dim,
        length,
    ) = hf_dataset_path

    hf_dataset = load_from_disk(dataset_path)
    indexer = HuggingFaceDatasetIndexer(hf_dataset)
    rng = np.random.default_rng(0)
    ts_lengths = [rng.integers(*length, endpoint=True) for _ in range(num_examples)]
    assert len(indexer) == num_examples
    item = indexer[0]
    _test_item(item, target_dim, past_feat_dynamic_real_dim, ts_lengths)


@pytest.mark.parametrize(
    "hf_dataset_path",
    [
        (
            3,
            1,
            None,
            (10, 10),
        ),  # num_examples, target_dim, past_feat_dynamic_real_dim, length
        (3, 1, None, (10, 20)),
        (3, 2, 1, (10, 10)),
        (3, 2, 5, (10, 20)),
    ],
    indirect=True,
)
def test_hf_dataset_indexer_list(hf_dataset_path):
    (
        dataset_path,
        num_examples,
        target_dim,
        past_feat_dynamic_real_dim,
        length,
    ) = hf_dataset_path

    hf_dataset = load_from_disk(dataset_path)
    indexer = HuggingFaceDatasetIndexer(hf_dataset)
    rng = np.random.default_rng(0)
    ts_lengths = [rng.integers(*length, endpoint=True) for _ in range(num_examples)]
    assert len(indexer) == num_examples
    list_batch = indexer[[0, 1]]
    _test_batch(list_batch, target_dim, past_feat_dynamic_real_dim, ts_lengths)


@pytest.mark.parametrize(
    "hf_dataset_path",
    [
        (
            3,
            1,
            None,
            (10, 10),
        ),  # num_examples, target_dim, past_feat_dynamic_real_dim, length
        (3, 1, None, (10, 20)),
        (3, 2, 1, (10, 10)),
        (3, 2, 5, (10, 20)),
    ],
    indirect=True,
)
def test_hf_dataset_indexer_slice(hf_dataset_path):
    (
        dataset_path,
        num_examples,
        target_dim,
        past_feat_dynamic_real_dim,
        length,
    ) = hf_dataset_path

    hf_dataset = load_from_disk(dataset_path)
    indexer = HuggingFaceDatasetIndexer(hf_dataset)
    rng = np.random.default_rng(0)
    ts_lengths = [rng.integers(*length, endpoint=True) for _ in range(num_examples)]
    assert len(indexer) == num_examples
    slice_batch = indexer[:2]
    _test_batch(slice_batch, target_dim, past_feat_dynamic_real_dim, ts_lengths)


@pytest.mark.parametrize(
    "hf_dataset_path",
    [
        (1, 2, 5, (10, 20)),
    ],
    indirect=True,
)
@pytest.mark.parametrize("idx", [1, [0, 1], slice(0, 2)])
def test_hf_dataset_indexer_index_error(
    hf_dataset_path,
    idx: int | list[int] | slice,
):
    (
        dataset_path,
        num_examples,
        target_dim,
        past_feat_dynamic_real_dim,
        length,
    ) = hf_dataset_path

    hf_dataset_indexer = HuggingFaceDatasetIndexer(load_from_disk(dataset_path))

    assert len(hf_dataset_indexer) == num_examples

    with pytest.raises(IndexError):
        hf_dataset_indexer[idx]
