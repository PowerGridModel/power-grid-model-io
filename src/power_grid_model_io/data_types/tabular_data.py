# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
The TabularData class is a wrapper around Dict[str, pd.DataFrame],
which supports unit conversions and value substitutions
"""

from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
import structlog

from power_grid_model_io.mappings.unit_mapping import UnitMapping
from power_grid_model_io.mappings.value_mapping import ValueMapping


class TabularData:
    """
    The TabularData class is a wrapper around Dict[str, pd.DataFrame],
    which supports unit conversions and value substitutions
    """

    def __init__(self, **tables: pd.DataFrame):
        for table_name, table_data in tables.items():
            if not isinstance(table_data, pd.DataFrame):
                raise TypeError(
                    f"Invalid data type for table '{table_name}'; "
                    f"expected a pandas DataFrame, got {type(table_data).__name__}."
                )
        self._data: Dict[str, pd.DataFrame] = tables
        self._units: Optional[UnitMapping] = None
        self._substitution: Optional[ValueMapping] = None
        self._log = structlog.get_logger(type(self).__name__)

    def set_unit_multipliers(self, units: UnitMapping):
        """
        Define unit multipliers.
        """
        self._units = units

    def set_substitutions(self, substitution: ValueMapping):
        """
        Define value substitutions
        """
        self._substitution = substitution

    def get_column(self, table: str, field: str) -> pd.Series:
        """
        Select a column from a table, while applying unit conversions and value substitutions
        """
        tbl_data = self._data[table]
        col_data = tbl_data[field]

        # If unit information is available, convert the unit
        if not isinstance(col_data, pd.Series):
            col_data = self._apply_unit_conversion(tbl_data=tbl_data, table=table, field=field)
            if not isinstance(col_data, pd.Series):
                raise TypeError(
                    f"The '{field}' column should now be unitless, "
                    f"but it still contains a unit: {col_data.columns.values}"
                )
        return self._apply_value_substitution(col_data=col_data, table=table, field=field)

    def _apply_value_substitution(self, col_data: pd.Series, table: str, field: str) -> pd.Series:

        if self._substitution is None:  # No subtitution defined, at all
            return col_data

        # Find substitutions, ignore if none is found
        try:
            substitutions = self._substitution.get_substitutions(field=f"{table}.{field}")
        except KeyError:
            try:
                substitutions = self._substitution.get_substitutions(field=field)
            except KeyError:
                return col_data

        if substitutions is None:  # No subtitution defined, for this column
            return col_data

        def sub(value):
            try:
                return substitutions[value]
            except KeyError as ex:
                raise KeyError(
                    f"Unknown substitution for value '{value}' in column '{field}' in table '{table}'"
                ) from ex

        # Apply substitutions
        self._log.debug("Apply value substitutions", table=table, field=field, substitutions=substitutions)
        return col_data.map(sub)

    def _apply_unit_conversion(self, tbl_data: pd.DataFrame, table: str, field: str) -> pd.Series:
        unit = tbl_data[field].columns[0]

        if unit:
            try:
                if self._units is None:
                    raise KeyError(unit)
                multiplier, si_unit = self._units.get_unit_multiplier(unit)
            except KeyError as ex:
                raise KeyError(f"Unknown unit '{unit}' for column '{field}' in table '{table}'") from ex
        else:
            si_unit = unit

        if unit == si_unit:
            self._log.debug("No unit conversion needed", table=table, field=field, unit=unit)
        else:
            self._log.debug(
                "Apply unit conversion", table=table, field=field, unit=unit, si_unit=si_unit, multiplier=multiplier
            )
            try:
                tbl_data[field] *= multiplier
            except TypeError as ex:
                self._log.warning(
                    f"The column '{field}' on table '{table}' does not seem to be numerical "
                    f"while trying to apply a multiplier ({multiplier}) for unit '{unit}': {ex}"
                )

            # Replace the unit with the SI unit
            tbl_data.columns = tbl_data.columns.values
            tbl_data.columns = pd.MultiIndex.from_tuples(tbl_data.rename(columns={(field, unit): (field, si_unit)}))

        return tbl_data[pd.MultiIndex.from_tuples([(field, si_unit)])[0]]

    def __contains__(self, table_name: str) -> bool:
        """
        Mimic the dictionary 'in' operator
        """
        return table_name in self._data

    def __getitem__(self, table_name: str):
        """
        Mimic the dictionary [] operator
        """
        return self._data[table_name]

    def keys(self) -> Iterable[str]:
        """
        Mimic the dictionary .keys() function
        """
        return self._data.keys()

    def items(self) -> Iterable[Tuple[str, pd.DataFrame]]:
        """
        Mimic the dictionary .items() function
        """
        return self._data.items()
