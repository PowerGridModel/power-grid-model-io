# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
Value substitution helper class
"""

import structlog

from power_grid_model_io.mappings.field_mapping import FieldMapping

Value = int | float | str | bool

#             attr      key    value
Values = dict[str, dict[Value, Value]]


# pylint: disable=too-few-public-methods
class ValueMapping(FieldMapping[dict[Value, Value]]):
    """
    Value substitution helper class
    """

    def __init__(self, mapping: Values | None = None, logger=None):
        super().__init__(mapping=mapping)
        if logger is None:
            self._log = structlog.get_logger(f"{__name__}_{id(self)}")
        else:
            self._log = logger
        if mapping is not None:
            self._log.debug(
                "Set value mapping", n_attributes=len(mapping), n_mappings=sum(len(m) for m in mapping.values())
            )

    def get_substitutions(self, attr: str, table: str | None = None) -> dict[Value, Value]:
        """
        Find the substitutions for a given attribute.
        """
        return self._get_mapping(attr=attr, table=table)
