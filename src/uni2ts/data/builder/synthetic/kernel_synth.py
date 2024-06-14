#  Copyright (c) 2024, Amazon.com, Inc. or its affiliates and Salesforce, Inc.
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
import functools
import os
from pathlib import Path
from typing import Optional

import datasets
import numpy as np
from datasets import Features, Sequence, Value
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    DotProduct,
    ExpSineSquared,
    Kernel,
    RationalQuadratic,
    WhiteKernel,
)
from torch.utils.data import Dataset

from uni2ts.common.env import env
from uni2ts.data.builder import DatasetBuilder
from uni2ts.data.dataset import SampleTimeSeriesType, TimeSeriesDataset
from uni2ts.data.indexer import HuggingFaceDatasetIndexer

LENGTH = 1024
KERNEL_BANK = [
    ExpSineSquared(periodicity=24 / LENGTH),  # H
    ExpSineSquared(periodicity=48 / LENGTH),  # 0.5H
    ExpSineSquared(periodicity=96 / LENGTH),  # 0.25H
    ExpSineSquared(periodicity=24 * 7 / LENGTH),  # H
    ExpSineSquared(periodicity=48 * 7 / LENGTH),  # 0.5H
    ExpSineSquared(periodicity=96 * 7 / LENGTH),  # 0.25H
    ExpSineSquared(periodicity=7 / LENGTH),  # D
    ExpSineSquared(periodicity=14 / LENGTH),  # 0.5D
    ExpSineSquared(periodicity=30 / LENGTH),  # D
    ExpSineSquared(periodicity=60 / LENGTH),  # 0.5D
    ExpSineSquared(periodicity=365 / LENGTH),  # D
    ExpSineSquared(periodicity=365 * 2 / LENGTH),  # 0.5D
    ExpSineSquared(periodicity=4 / LENGTH),  # W
    ExpSineSquared(periodicity=26 / LENGTH),  # W
    ExpSineSquared(periodicity=52 / LENGTH),  # W
    ExpSineSquared(periodicity=4 / LENGTH),  # M
    ExpSineSquared(periodicity=6 / LENGTH),  # M
    ExpSineSquared(periodicity=12 / LENGTH),  # M
    ExpSineSquared(periodicity=4 / LENGTH),  # Q
    ExpSineSquared(periodicity=4 * 10 / LENGTH),  # Q
    ExpSineSquared(periodicity=10 / LENGTH),  # Y
    DotProduct(sigma_0=0.0),
    DotProduct(sigma_0=1.0),
    DotProduct(sigma_0=10.0),
    RBF(length_scale=0.1),
    RBF(length_scale=1.0),
    RBF(length_scale=10.0),
    RationalQuadratic(alpha=0.1),
    RationalQuadratic(alpha=1.0),
    RationalQuadratic(alpha=10.0),
    WhiteKernel(noise_level=0.1),
    WhiteKernel(noise_level=1.0),
    ConstantKernel(),
]


def random_binary_map(a: Kernel, b: Kernel):
    """
    Applies a random binary operator (+ or *) with equal probability
    on kernels ``a`` and ``b``.

    Parameters
    ----------
    a
        A GP kernel.
    b
        A GP kernel.

    Returns
    -------
        The composite kernel `a + b` or `a * b`.
    """
    binary_maps = [lambda x, y: x + y, lambda x, y: x * y]
    return np.random.choice(binary_maps)(a, b)


def sample_from_gp_prior(
    kernel: Kernel, X: np.ndarray, random_seed: Optional[int] = None
):
    """
    Draw a sample from a GP prior.

    Parameters
    ----------
    kernel
        The GP covaraince kernel.
    X
        The input "time" points.
    random_seed, optional
        The random seed for sampling, by default None.

    Returns
    -------
        A time series sampled from the GP prior.
    """
    if X.ndim == 1:
        X = X[:, None]

    assert X.ndim == 2
    gpr = GaussianProcessRegressor(kernel=kernel)
    ts = gpr.sample_y(X, n_samples=1, random_state=random_seed)

    return ts


def sample_from_gp_prior_efficient(
    kernel: Kernel,
    X: np.ndarray,
    random_seed: Optional[int] = None,
    method: str = "eigh",
):
    """
    Draw a sample from a GP prior. An efficient version that allows specification
    of the sampling method. The default sampling method used in GaussianProcessRegressor
    is based on SVD which is significantly slower that alternatives such as `eigh` and
    `cholesky`.

    Parameters
    ----------
    kernel
        The GP covaraince kernel.
    X
        The input "time" points.
    random_seed, optional
        The random seed for sampling, by default None.
    method, optional
        The sampling method for multivariate_normal, by default `eigh`.

    Returns
    -------
        A time series sampled from the GP prior.
    """
    if X.ndim == 1:
        X = X[:, None]

    assert X.ndim == 2

    cov = kernel(X)
    ts = np.random.default_rng(seed=random_seed).multivariate_normal(
        mean=np.zeros(X.shape[0]), cov=cov, method=method
    )

    return ts


def generate_time_series(max_kernels: int = 5):
    """Generate a synthetic time series from KernelSynth.

    Parameters
    ----------
    max_kernels, optional
        The maximum number of base kernels to use for each time series, by default 5

    Returns
    -------
        A time series generated by KernelSynth.
    """
    while True:
        X = np.linspace(0, 1, LENGTH)

        # Randomly select upto max_kernels kernels from the KERNEL_BANK
        selected_kernels = np.random.choice(
            KERNEL_BANK, np.random.randint(1, max_kernels + 1), replace=True
        )

        # Combine the sampled kernels using random binary operators
        kernel = functools.reduce(random_binary_map, selected_kernels)

        # Sample a time series from the GP prior
        try:
            ts = sample_from_gp_prior(kernel=kernel, X=X)
        except np.linalg.LinAlgError as err:
            print("Error caught:", err)
            continue

        # The timestamp is arbitrary
        return {"start": np.datetime64("2000-01-01 00:00", "s"), "target": ts.squeeze()}


class KernelSynthDatasetBuilder(DatasetBuilder):
    def __init__(
        self,
        freq: str,
        weight: float = 1.0,
        storage_path: Path = env.CUSTOM_DATA_PATH,
    ):
        self.freq = freq
        self.weight = weight
        self.storage_path = Path(storage_path)
        self.dataset_name = f"kernel_synth_{self.freq}"

    def build_dataset(
        self, num_series: int, max_kernels: int, num_proc: int = os.cpu_count()
    ):
        def example_gen_func(shards: list[tuple[int, int]]):
            for start, end in shards:
                for idx in range(start, end):
                    time_series = generate_time_series(max_kernels)
                    yield time_series | dict(item_id=f"item_{idx}", freq=self.freq)

        features = Features(
            dict(
                item_id=Value("string"),
                start=Value("timestamp[s]"),
                freq=Value("string"),
                target=Sequence(Value("float32")),
            )
        )
        shards = [
            (idx * num_series // num_proc, (idx + 1) * num_series // num_proc)
            for idx in range(num_proc)
        ]
        hf_dataset = datasets.Dataset.from_generator(
            example_gen_func,
            features=features,
            gen_kwargs={"shards": shards},
            num_proc=num_proc,
            cache_dir=env.HF_CACHE_PATH,
        )
        hf_dataset.info.dataset_name = self.dataset_name
        hf_dataset.save_to_disk(self.storage_path / self.dataset_name)

    def load_dataset(self, transform_map: dict) -> Dataset:
        return TimeSeriesDataset(
            HuggingFaceDatasetIndexer(
                datasets.load_from_disk(str(self.storage_path / self.dataset_name))
            ),
            transform=transform_map[self.dataset_name](),
            dataset_weight=self.weight,
            sample_time_series=SampleTimeSeriesType.PROPORTIONAL,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-F", "--freq", type=str, default="H")
    parser.add_argument("-N", "--num-series", type=int, default=1000_000)
    parser.add_argument("-J", "--max-kernels", type=int, default=5)
    args = parser.parse_args()

    KernelSynthDatasetBuilder(freq=args.freq).build_dataset(
        num_series=args.num_series,
        max_kernels=args.max_kernels,
    )
