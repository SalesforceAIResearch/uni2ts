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
from typing import Any

from hydra.utils import get_class
from omegaconf import OmegaConf


def register_resolver(name: str) -> Callable[[Callable], Callable]:
    def decorator(resolver: Callable) -> Callable:
        OmegaConf.register_new_resolver(name, resolver)
        return resolver

    return decorator


@register_resolver("as_tuple")
def resolve_as_tuple(ls: list) -> tuple:
    return tuple(ls)


@register_resolver("cls_getattr")
def resolve_cls_getattr(cls_name: str, attribute_name: str) -> Any:
    if cls_name.endswith(".load_from_checkpoint"):
        cls_name = cls_name[: -len(".load_from_checkpoint")]
    cls = get_class(cls_name)
    return getattr(cls, attribute_name)


@register_resolver("floordiv")
def resolve_floordiv(a: int, b: int) -> int:
    return a // b


@register_resolver("mul")
def resolve_mul(a: float, b: float) -> float:
    return a * b
