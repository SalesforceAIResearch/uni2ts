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

import pytest

# Add the path of any pytest fixture files you want to make global
pytest_plugins = [
    # "test.fixture.autouse_fixture",
    "test.fixture.fixture",
]


def pytest_addoption(parser):
    parser.addoption(
        "--run-slow", action="store_true", default=False, help="run slow tests"
    )
    parser.addoption("--all", action="store_true", default=False, help="run all tests")


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow to run")


def pytest_collection_modifyitems(config, items):
    if config.getoption("--all"):
        # --all given in cli: do not skip any test
        return

    markers = {}
    if not config.getoption("--run-slow"):
        markers["slow"] = pytest.mark.skip(reason="need --run-slow option to run")

    for item in items:
        for key, marker in markers.items():
            if key in item.keywords:
                item.add_marker(marker)
