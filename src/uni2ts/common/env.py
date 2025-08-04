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
    """
    Retrieves an environment variable and converts it to a Path object.

    Args:
        var (Optional[str]): The name of the environment variable to retrieve.

    Returns:
        Optional[Path]: A Path object representing the value of the environment variable,
                        or None if the variable is not set.
    """
    if (path := os.getenv(var)) is not None:
        return Path(path)
    return None


class Env:
    """
    A singleton class to manage environment variables for the project.
    It loads variables from a .env file and provides them as class attributes.
    Path-related variables are automatically converted to Path objects.

    Attributes:
        path_vars (list[str]): A list of environment variable names that should be
                               treated as paths.
    """
    _instance: Optional["Env"] = None
    path_vars: list[str] = [
        "LOTSA_V1_PATH",
        "LSF_PATH",
        "CUSTOM_DATA_PATH",
        "HF_CACHE_PATH",
    ]

    def __new__(cls):
        """
        Creates a new instance of the Env class if one does not already exist.
        This ensures that the .env file is loaded only once.
        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            if not load_dotenv():
                warnings.warn("Failed to load .env file.")
            cls.monkey_patch_path_vars()
        return cls._instance

    @classmethod
    def monkey_patch_path_vars(cls):
        """
        Iterates through the `path_vars` list and sets a class attribute for each
        variable with the corresponding value from the environment. The value is
        retrieved and converted to a Path object using `get_path_var`.
        """
        for var in cls.path_vars:
            setattr(cls, var, get_path_var(var))


env = Env()
