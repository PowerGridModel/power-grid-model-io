# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
Unit mapping helper class
"""

from numbers import Number
from typing import Dict, Optional, Set, Tuple

import numpy as np
import structlog

#            si-unit   unit factor
Units = Dict[str, Optional[Dict[str, float]]]


class UnitMapping:
    """
    Unit mapping helper class.
    The input data is expected to be of the form:
    {
        "A" : None,
        "W": {
            "kW": 1000.0,
            "MW": 1000000.0
        }
     }
    """

    def __init__(self, mapping: Optional[Units] = None, logger=None):
        if logger is None:
            self._log = structlog.get_logger(f"{__name__}_{id(self)}")
        else:
            self._log = logger
        self._si_units: Set[str] = set()
        self._mapping: Dict[str, Tuple[float, str]] = {}
        self.set_mapping(mapping=mapping if mapping is not None else {})

    def set_mapping(self, mapping: Units):
        """
        Creates an internal mapping lookup table based on input data of the form:
        mapping = {
            "A" : None,
            "W": {
                "kW": 1000.0,
                "MW": 1000000.0
            }
         }
        """
        self._si_units = set(mapping.keys())
        self._mapping = {}
        for si_unit, multipliers in mapping.items():
            if not multipliers:
                continue
            for unit, multiplier in multipliers.items():
                if unit in self._mapping:
                    multiplier_, si_unit_ = self._mapping[unit]
                    raise ValueError(
                        f"Multiple unit definitions for '{unit}': "
                        f"1{unit} = {multiplier_}{si_unit_} = {multiplier}{si_unit}"
                    )
                self._mapping[unit] = (multiplier, si_unit)
                if unit == si_unit:
                    if not np.isclose(multiplier, 1.0, rtol=1.0e-9, atol=0.0):
                        raise ValueError(
                            f"Invalid unit definition for '{unit}': 1{unit} cannot be {multiplier}{si_unit}"
                        )
                    continue  # pragma: no cover (bug in python 3.9)
                self._mapping[unit] = (multiplier, si_unit)
        self._log.debug(
            "Set unit definitions", n_units=len(self._si_units | self._mapping.keys()), n_si_units=len(self._si_units)
        )

    def get_unit_multiplier(self, unit: str) -> Tuple[float, str]:
        """
        Find the correct unit multiplier and the corresponding SI unit
        """
        if unit in self._si_units:
            return 1.0, unit

        multiplier, si_unit = self._mapping[unit]

        if not isinstance(multiplier, Number):
            raise TypeError(f"The multiplier ({multiplier}) for unit '{unit}' is not numerical.")

        return multiplier, si_unit
