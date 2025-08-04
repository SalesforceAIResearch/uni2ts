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

import abc
from typing import Any, Callable

from torch.utils.data import ConcatDataset, Dataset

from uni2ts.transform import Transformation


# TODO: Add __repr__
class DatasetBuilder(abc.ABC):
    """
    An abstract base class for dataset builders. It defines a common interface for
    building and loading datasets.
    """

    @abc.abstractmethod
    def build_dataset(self, *args, **kwargs):
        """
        An abstract method for building a dataset. This method should handle the
        logic for creating a dataset in the required file format.
        """
        ...

    @abc.abstractmethod
    def load_dataset(
        self, transform_map: dict[Any, Callable[..., Transformation]]
    ) -> Dataset:
        """
        An abstract method for loading a dataset.

        Args:
            transform_map (dict[Any, Callable[..., Transformation]]): A dictionary
                mapping keys to transformation functions. This allows for applying
                different transformations to different parts of the dataset.

        Returns:
            Dataset: A PyTorch Dataset object ready for training.
        """
        ...


class ConcatDatasetBuilder(DatasetBuilder):
    """
    A dataset builder that concatenates multiple `DatasetBuilder` instances. This allows
    for combining multiple datasets into a single dataset.

    Args:
        *builders (DatasetBuilder): A variable number of `DatasetBuilder` instances
            to concatenate.
    """

    def __init__(self, *builders: DatasetBuilder):
        super().__init__()
        assert len(builders) > 0, "Must provide at least one builder to ConcatBuilder"
        assert all(
            isinstance(builder, DatasetBuilder) for builder in builders
        ), "All builders must be instances of DatasetBuilder"
        self.builders: tuple[DatasetBuilder, ...] = builders

    def build_dataset(self):
        """
        This method is not implemented for `ConcatDatasetBuilder`, as it is intended
        to combine already built datasets.
        """
        raise ValueError(
            "Do not use ConcatBuilder to build datasets, build sub datasets individually instead."
        )

    def load_dataset(
        self, transform_map: dict[Any, Callable[..., Transformation]]
    ) -> ConcatDataset:
        """
        Loads all the datasets from the builders and concatenates them using
        `torch.utils.data.ConcatDataset`.

        Args:
            transform_map (dict[Any, Callable[..., Transformation]]): A dictionary
                mapping keys to transformation functions.

        Returns:
            ConcatDataset: A concatenated dataset.
        """
        return ConcatDataset(
            [builder.load_dataset(transform_map) for builder in self.builders]
        )
