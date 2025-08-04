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
from typing import TypeVar

T = TypeVar("T")


def abstract_class_property(*names: str) -> Callable[[type[T], ...], type[T]]:
    """
    A class decorator that ensures specified class-level properties are implemented by subclasses.
    This is useful for creating abstract base classes with required class-level attributes that
    are not part of the standard ABC mechanism, which primarily focuses on methods.

    Args:
        *names (str): A variable number of strings, where each string is the name of a
                      class property that must be implemented by any subclass.

    Returns:
        Callable[[type[T], ...], type[T]]: A decorator that takes a class and returns
                                             the same class with a modified __init_subclass__
                                             method to enforce property implementation.

    Raises:
        NotImplementedError: If a subclass does not implement one of the specified properties,
                             or if the property is set to NotImplemented.
    """
    def _func(cls: type[T]) -> type[T]:
        # Store the original __init_subclass__ method to be called later.
        original_init_subclass = cls.__init_subclass__

        def _init_subclass(_cls, **kwargs):
            """
            This method is called whenever a class inheriting from the decorated class is defined.
            It first calls the original __init_subclass__ and then checks for the presence and
            implementation of the required class properties.
            """
            # The default implementation of __init_subclass__ takes no
            # positional arguments, but a custom implementation does.
            # If the user has not reimplemented __init_subclass__ then
            # the first signature will fail and we try the second.
            try:
                original_init_subclass(_cls, **kwargs)
            except TypeError:
                original_init_subclass(**kwargs)

            # Check that each attribute is defined and implemented.
            for name in names:
                if not hasattr(_cls, name):
                    raise NotImplementedError(
                        f"Class property '{name}' has not been defined for {_cls.__name__}"
                    )
                if getattr(_cls, name, NotImplemented) is NotImplemented:
                    raise NotImplementedError(
                        f"Class property '{name}' has not been implemented for {_cls.__name__}"
                    )
        
        # Replace the class's __init_subclass__ with the new version.
        cls.__init_subclass__ = classmethod(_init_subclass)
        return cls

    return _func
