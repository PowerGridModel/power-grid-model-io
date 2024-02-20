# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
Value substitution helper class
"""
from typing import Dict, Optional, Union

import structlog

from power_grid_model_io.mappings.field_mapping import FieldMapping

Value = Union[int, float, str, bool]

#             attr      key    value
Values = Dict[str, Dict[Value, Value]]


# pylint: disable=too-few-public-methods
class ValueMapping(FieldMapping[Dict[Value, Value]]):
    """
    Value substitution helper class
    """

    def __init__(self, mapping: Optional[Values] = None, logger=None):
        super().__init__(mapping=mapping)
        if logger is None:
            self._log = structlog.get_logger(f"{__name__}_{id(self)}")
        else:
            self._log = logger
        if mapping is not None:
            self._log.debug(
                "Set value mapping", n_attributes=len(mapping), n_mappings=sum(len(m) for m in mapping.values())
            )

    def get_substitutions(self, attr: str, table: Optional[str] = None) -> Dict[Value, Value]:
        """
        Find the substitutions for a given attribute.
        """
        return self._get_mapping(attr=attr, table=table)
