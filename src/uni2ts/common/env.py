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

import os
import warnings
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv


def get_path_var(var: Optional[str]) -> Optional[Path]:
    if (path := os.getenv(var)) is not None:
        return Path(path)
    return None


class Env:
    _instance: Optional["Env"] = None
    path_vars: list[str] = [
        "LOTSA_V1_PATH",
        "LSF_PATH",
        "CUSTOM_DATA_PATH",
        "HF_CACHE_PATH",
    ]

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            if not load_dotenv():
                warnings.warn("Failed to load .env file.")
            cls.monkey_patch_path_vars()
        return cls._instance

    @classmethod
    def monkey_patch_path_vars(cls):
        for var in cls.path_vars:
            setattr(cls, var, get_path_var(var))


env = Env()
