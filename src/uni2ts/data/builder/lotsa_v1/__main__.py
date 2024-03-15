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

parser = argparse.ArgumentParser()
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
parser.add_argument(
    "--datasets",
    type=str,
    nargs="+",
    default=None,
    help="The datasets to generate",
)
parser.add_argument(
    "--storage_path",
    type=Path,
    default=env.LOTSA_V1_PATH,
    help="Path of directory to save the datasets",
)
parser.add_argument(
    "--overwrite",
    action="store_true",
)
args = parser.parse_args()

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

datasets = set(args.datasets or Builder.dataset_list)
found = {directory.stem for directory in args.storage_path.iterdir()}
overlap = datasets & found

if len(overlap) > 0:
    print(f"Found datasets already present in storage path: {overlap}")

if not args.overwrite:
    datasets = datasets - found
    if len(overlap) > 0:
        print(f"Skipping processed datasets, building: {list(datasets)}")
        print("To overwrite existing datasets, use the `--overwrite` flag")
else:
    print(f"Overwriting existing datasets, building: {datasets}")

failed = {}
for dataset in datasets:
    try:
        print(f"Building: {dataset}")
        Builder(
            datasets=list(datasets),
            storage_path=args.storage_path,
        ).build_dataset(dataset=dataset)
        print(f"Successfully built {dataset}")
    except Exception as e:
        print(f"Failed to build {dataset}")
        failed[dataset] = traceback.format_exc()

if len(failed) > 0:
    print(f"Failed: {list(failed.keys())}")
    for k, v in failed.items():
        print(f"{k}: {v}")
