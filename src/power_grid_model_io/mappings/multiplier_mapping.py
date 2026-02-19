# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
Field multiplier helper class
"""

from power_grid_model_io.mappings.field_mapping import FieldMapping

Multipliers = dict[str, float]


# pylint: disable=too-few-public-methods
class MultiplierMapping(FieldMapping[float]):
    """
    Field multiplier helper class
    """

    def __init__(self, mapping: Multipliers | None = None, logger=None):
        super().__init__(mapping=mapping, logger=logger)
        if mapping is not None:
            self._log.debug("Set multiplier mapping", n_attributes=len(mapping))

    def get_multiplier(self, attr: str, table: str | None = None) -> float:
        """
        Find the multiplier for a given attribute.
        """
        return self._get_mapping(attr=attr, table=table)
