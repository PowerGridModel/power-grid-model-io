# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
RegEx attribute based mapping class
"""

import re
from functools import lru_cache
from typing import Dict, Generic, Optional, TypeVar

import structlog

T = TypeVar("T")


# pylint: disable=too-few-public-methods
class FieldMapping(Generic[T]):
    """
    RegEx attribute based mapping class
    """

    def __init__(self, mapping: Optional[Dict[str, T]] = None):
        self._log = structlog.get_logger(type(self).__name__)
        self._values: Dict[str, T] = mapping or {}
        self._values_re: Dict[re.Pattern, T] = {re.compile(pattern): value for pattern, value in self._values.items()}

    @lru_cache
    def _get_mapping(self, attr: str, table: Optional[str] = None) -> T:
        """
        Find the mapping for a given attribute.
        """

        # If a table is supplied, first try to find a table specific match
        keys = []
        if table is not None:
            keys.append(f"{table}.{attr}")
        keys.append(attr)

        # First check if there is an exact match
        for key in keys:
            if key in self._values:
                return self._values[key]

        # Otherwise, use the values as regular expressions
        for key in keys:
            for pattern in self._values_re.keys():
                if pattern.fullmatch(key):
                    return self._values_re[pattern]

        # If no match was found, raise a key error
        raise KeyError(keys.pop())
