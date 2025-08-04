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
import traceback
from pathlib import Path

from uni2ts.common.env import env

from . import (
    Buildings900KDatasetBuilder,
    BuildingsBenchDatasetBuilder,
    CloudOpsTSFDatasetBuilder,
    CMIP6DatasetBuilder,
    ERA5DatasetBuilder,
    GluonTSDatasetBuilder,
    LargeSTDatasetBuilder,
    LibCityDatasetBuilder,
    OthersLOTSADatasetBuilder,
    ProEnFoDatasetBuilder,
    SubseasonalDatasetBuilder,
)

# Create an argument parser to handle command-line arguments.
parser = argparse.ArgumentParser()
# Add an argument for the builder name, with a limited set of choices.
parser.add_argument(
    "builder",
    type=str,
    choices=[
        "buildings_900k",
        "buildings_bench",
        "cloudops_tsf",
        "cmip6",
        "era5",
        "gluonts",
        "largest",
        "lib_city",
        "others",
        "proenfo",
        "subseasonal",
    ],
)
# Add an optional argument for specifying which datasets to generate.
parser.add_argument(
    "--datasets",
    type=str,
    nargs="+",
    default=None,
    help="The datasets to generate",
)
# Add an optional argument for the storage path, with a default value from the environment.
parser.add_argument(
    "--storage_path",
    type=Path,
    default=env.LOTSA_V1_PATH,
    help="Path of directory to save the datasets",
)
# Add an optional flag to allow overwriting existing datasets.
parser.add_argument(
    "--overwrite",
    action="store_true",
)
# Parse the command-line arguments.
args = parser.parse_args()

# A dictionary mapping builder names to their corresponding classes.
Builder = {
    "buildings_900k": Buildings900KDatasetBuilder,
    "buildings_bench": BuildingsBenchDatasetBuilder,
    "cloudops_tsf": CloudOpsTSFDatasetBuilder,
    "cmip6": CMIP6DatasetBuilder,
    "era5": ERA5DatasetBuilder,
    "gluonts": GluonTSDatasetBuilder,
    "largest": LargeSTDatasetBuilder,
    "lib_city": LibCityDatasetBuilder,
    "others": OthersLOTSADatasetBuilder,
    "proenfo": ProEnFoDatasetBuilder,
    "subseasonal": SubseasonalDatasetBuilder,
}[args.builder]

# Determine the set of datasets to build.
datasets = set(args.datasets or Builder.dataset_list)
# Find which datasets already exist in the storage path.
found = {directory.stem for directory in args.storage_path.iterdir()}
overlap = datasets & found

# If there are overlapping datasets, print a message.
if len(overlap) > 0:
    print(f"Found datasets already present in storage path: {overlap}")

# If not overwriting, remove the existing datasets from the set to be built.
if not args.overwrite:
    datasets = datasets - found
    if len(overlap) > 0:
        print(f"Skipping processed datasets, building: {list(datasets)}")
        print("To overwrite existing datasets, use the `--overwrite` flag")
else:
    print(f"Overwriting existing datasets, building: {datasets}")

# A dictionary to store any failed dataset builds.
failed = {}
# Iterate over the datasets to be built.
for dataset in datasets:
    try:
        print(f"Building: {dataset}")
        # Instantiate the builder and build the dataset.
        Builder(
            datasets=list(datasets),
            storage_path=args.storage_path,
        ).build_dataset(dataset=dataset)
        print(f"Successfully built {dataset}")
    except Exception as e:
        # If an exception occurs, record the failure.
        print(f"Failed to build {dataset}")
        failed[dataset] = traceback.format_exc()

# If there were any failures, print a summary.
if len(failed) > 0:
    print(f"Failed: {list(failed.keys())}")
    for k, v in failed.items():
        print(f"{k}: {v}")
