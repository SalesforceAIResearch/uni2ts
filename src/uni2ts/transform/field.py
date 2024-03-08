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

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ._base import Transformation


@dataclass
class SetValue:
    value: Any

    def __call__(self, data_entry: dict[str, Any]) -> Any:
        return self.value


@dataclass
class LambdaSetFieldIfNotPresent(Transformation):
    field: str
    get_value: Callable[[dict[str, Any]], Any]

    @staticmethod
    def set_field(data_entry: dict[str, Any], field: str, value: Any) -> dict[str, Any]:
        if field not in data_entry.keys():
            data_entry[field] = value
        return data_entry

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        return self.set_field(data_entry, self.field, self.get_value(data_entry))


@dataclass
class SelectFields(Transformation):
    fields: list[str]
    allow_missing: bool = False

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        if self.allow_missing:
            return {f: data_entry[f] for f in self.fields if f in data_entry}
        return {f: data_entry[f] for f in self.fields}


@dataclass
class RemoveFields(Transformation):
    fields: list[str]

    def __call__(self, data_entry: dict[str, Any]) -> dict[str, Any]:
        for k in self.fields:
            data_entry.pop(k, None)
        return data_entry
