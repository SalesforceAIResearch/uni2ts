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
    Base class for DatasetBuilders.
    """

    @abc.abstractmethod
    def build_dataset(self, *args, **kwargs):
        """
        Builds the dataset into the required file format.
        """
        ...

    @abc.abstractmethod
    def load_dataset(
        self, transform_map: dict[Any, Callable[..., Transformation]]
    ) -> Dataset:
        """
        Load the dataset.

        :param transform_map: a map which returns the required dataset transformations to be applied
        :return: the dataset ready for training
        """
        ...


class ConcatDatasetBuilder(DatasetBuilder):
    """
    Concatenates DatasetBuilders such that they can be loaded together.
    """

    def __init__(self, *builders: DatasetBuilder):
        """
        :param builders: DatasetBuilders to be concatenated together.
        """
        super().__init__()
        assert len(builders) > 0, "Must provide at least one builder to ConcatBuilder"
        assert all(
            isinstance(builder, DatasetBuilder) for builder in builders
        ), "All builders must be instances of DatasetBuilder"
        self.builders: tuple[DatasetBuilder, ...] = builders

    def build_dataset(self):
        raise ValueError(
            "Do not use ConcatBuilder to build datasets, build sub datasets individually instead."
        )

    def load_dataset(
        self, transform_map: dict[Any, Callable[..., Transformation]]
    ) -> ConcatDataset:
        """
        Loads all builders with ConcatDataset.

        :param transform_map: a map which returns the required dataset transformations to be applied
        :return: the dataset ready for training
        """
        return ConcatDataset(
            [builder.load_dataset(transform_map) for builder in self.builders]
        )
