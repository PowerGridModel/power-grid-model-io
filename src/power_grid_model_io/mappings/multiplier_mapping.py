# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Field multiplier helper class
"""

from typing import Dict, Optional

from power_grid_model_io.mappings.field_mapping import FieldMapping

Multipliers = Dict[str, float]


# pylint: disable=too-few-public-methods
class MultiplierMapping(FieldMapping[float]):
    """
    Field multiplier helper class
    """

    def __init__(self, mapping: Optional[Multipliers] = None):
        super().__init__(mapping=mapping)
        if mapping is not None:
            self._log.debug("Set multiplier mapping", n_attributes=len(mapping))

    def get_multiplier(self, attr: str, table: Optional[str] = None) -> float:
        """
        Find the multiplier for a given attribute.
        """
        return self._get_mapping(attr=attr, table=table)
