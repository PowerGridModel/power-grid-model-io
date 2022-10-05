# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Value substitution helper class
"""

import re
from functools import lru_cache
from typing import Dict, Optional, Union

import structlog

Value = Union[int, float, str, bool]

#             attr      key    value
Values = Dict[str, Dict[Value, Value]]
ValuesRe = Dict[re.Pattern, Dict[Value, Value]]


# pylint: disable=too-few-public-methods
class ValueMapping:
    """
    Value substitution helper class
    """

    def __init__(self, mapping: Optional[Values] = None):
        self._log = structlog.get_logger(type(self).__name__)
        self._values: Values = mapping or {}
        self._values_re: ValuesRe = {re.compile(pattern): value for pattern, value in self._values.items()}
        if mapping is not None:
            self._log.debug(
                "Set value mapping", n_fields=len(mapping), n_mappings=sum(len(m) for m in mapping.values())
            )

    @lru_cache
    def get_substitutions(self, field: str) -> Dict[Value, Value]:
        """
        Find the substitutions for a given field.
        """

        # First check if there is an exact match
        if field in self._values:
            return self._values[field]

        # Otherwise, use the values as regular expressions
        for pattern in self._values_re.keys():
            if pattern.fullmatch(field):
                return self._values_re[pattern]
        raise KeyError(field)
