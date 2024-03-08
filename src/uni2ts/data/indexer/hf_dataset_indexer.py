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

from collections.abc import Iterable

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
from datasets import Dataset
from datasets.features import Sequence
from datasets.formatting import query_table

from uni2ts.common.typing import BatchedData, Data, MultivarTimeSeries, UnivarTimeSeries

from ._base import Indexer


class HuggingFaceDatasetIndexer(Indexer):
    def __init__(self, dataset: Dataset, uniform: bool = False):
        super().__init__(uniform=uniform)
        self.dataset = dataset
        self.features = dict(self.dataset.features)
        self.non_seq_cols = [
            name
            for name, feat in self.features.items()
            if not isinstance(feat, Sequence)
        ]
        self.seq_cols = [
            name for name, feat in self.features.items() if isinstance(feat, Sequence)
        ]
        self.dataset.set_format("numpy", columns=self.non_seq_cols)

    def __len__(self) -> int:
        return len(self.dataset)

    def _getitem_int(self, idx: int) -> dict[str, Data]:
        non_seqs = self.dataset[idx]
        pa_subtable = query_table(self.dataset.data, idx, indices=self.dataset._indices)
        seqs = {
            col: self._pa_column_to_numpy(pa_subtable, col)[0] for col in self.seq_cols
        }
        return non_seqs | seqs

    def _getitem_iterable(self, idx: Iterable[int]) -> dict[str, BatchedData]:
        non_seqs = self.dataset[idx]
        pa_subtable = query_table(self.dataset.data, idx, indices=self.dataset._indices)
        seqs = {
            col: self._pa_column_to_numpy(pa_subtable, col) for col in self.seq_cols
        }
        return non_seqs | seqs

    def _getitem_slice(self, idx: slice) -> dict[str, BatchedData]:
        non_seqs = self.dataset[idx]
        pa_subtable = query_table(self.dataset.data, idx, indices=self.dataset._indices)
        seqs = {
            col: self._pa_column_to_numpy(pa_subtable, col) for col in self.seq_cols
        }
        return non_seqs | seqs

    def _pa_column_to_numpy(
        self, pa_table: pa.Table, column_name: str
    ) -> list[UnivarTimeSeries] | list[MultivarTimeSeries]:
        pa_array: pa.Array = pa_table.column(column_name)
        feature = self.features[column_name]

        if isinstance(pa_array, pa.ChunkedArray):
            if isinstance(feature.feature, Sequence):
                array = [
                    flat_slice.flatten().to_numpy(False).reshape(feat_length, -1)
                    for chunk in pa_array.chunks
                    for i in range(len(chunk))
                    if (flat_slice := chunk.slice(i, 1).flatten())
                    and (
                        feat_length := (
                            feature.length if feature.length != -1 else len(flat_slice)
                        )
                    )
                ]
            else:
                array = [
                    chunk.slice(i, 1).flatten().to_numpy(False)
                    for chunk in pa_array.chunks
                    for i in range(len(chunk))
                ]
        elif isinstance(pa_array, pa.ListArray):
            if isinstance(feature.feature, Sequence):
                flat_slice = pa_array.flatten()
                feat_length = (
                    feature.length if feature.length != -1 else len(flat_slice)
                )
                array = [flat_slice.flatten().to_numpy(False).reshape(feat_length, -1)]
            else:
                array = [pa_array.flatten().to_numpy(False)]
        else:
            raise NotImplementedError

        return array

    def get_proportional_probabilities(self, field: str = "target") -> np.ndarray:
        if self.uniform:
            return self.get_uniform_probabilities()

        if self[0]["target"].ndim > 1:
            lengths = pc.list_value_length(
                pc.list_flatten(pc.list_slice(self.dataset.data.column(field), 0, 1))
            )
        else:
            lengths = pc.list_value_length(self.dataset.data.column(field))
        lengths = lengths.to_numpy()
        probs = lengths / lengths.sum()
        return probs
