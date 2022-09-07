# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
from typing import Dict, Set, Tuple

import structlog

#            si-unit   unit factor
Units = Dict[str, Dict[str, float]]


class UnitMapping:
    def __init__(self, mapping: Units):
        self._log = structlog.get_logger(type(self).__name__)
        self._si_units: Set[str] = set()
        self._mapping: Dict[str, Tuple[float, str]] = {}
        self.set_mapping(mapping)

    def set_mapping(self, mapping: Units):
        self._si_units = set(mapping.keys())
        self._mapping = {}
        for si_unit, multipliers in mapping.items():
            if not multipliers:
                continue
            for unit, multiplier in multipliers.items():
                if unit in self._mapping:
                    multiplier_, si_unit_ = self._mapping[unit]
                    raise ValueError(
                        f"Multiple mapping for {unit}; " f"1 {unit} = {multiplier_} {si_unit_} = {multiplier} {si_unit}"
                    )
                self._mapping[unit] = (multiplier, si_unit)
            if unit == si_unit:
                if multiplier != 1.0:
                    raise ValueError(f"Invalid mapping for {unit}; " f"1 {unit} = {multiplier} {si_unit}")
                continue
            self._mapping[unit] = (multiplier, si_unit)
        self._log.debug(
            f"Set unit definitions", n_units=len(self._si_units | self._mapping.keys()), n_si_units=len(self._si_units)
        )

    def get_unit_multiplier(self, unit: str) -> Tuple[float, str]:
        return (1.0, unit) if unit in self._si_units else self._mapping[unit]

    def to_si_unit(self, value: float, unit: str) -> Tuple[float, str]:
        if unit in self._si_units:
            return value, unit
        multiplier, unit = self._mapping[unit]
        return value, unit
