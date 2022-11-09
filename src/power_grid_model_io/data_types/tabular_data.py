# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
The TabularData class is a wrapper around Dict[str, Union[pd.DataFrame, np.ndarray]],
which supports unit conversions and value substitutions
"""

from typing import Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd
import structlog

from power_grid_model_io.mappings.unit_mapping import UnitMapping
from power_grid_model_io.mappings.value_mapping import ValueMapping


class TabularData:
    """
    The TabularData class is a wrapper around Dict[str, Union[pd.DataFrame, np.ndarray]],
    which supports unit conversions and value substitutions
    """

    def __init__(self, **tables: Union[pd.DataFrame, np.ndarray]):
        for table_name, table_data in tables.items():
            if not isinstance(table_data, (pd.DataFrame, np.ndarray)):
                raise TypeError(
                    f"Invalid data type for table '{table_name}'; "
                    f"expected a pandas DataFrame or NumPy array, got {type(table_data).__name__}."
                )
        self._data: Dict[str, Union[pd.DataFrame, np.ndarray]] = tables
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

    def get_column(self, table_name: str, column_name: str) -> pd.Series:
        """
        Select a column from a table, while applying unit conversions and value substitutions
        """
        table_data = self._data[table_name]

        # If the index 'column' is requested, but no column called 'index' exist,
        # return the index of the dataframe as if it were an actual column.
        if column_name == "index" and column_name not in table_data:
            return pd.Series(table_data.index, name="index")

        column_data = table_data[column_name]

        if isinstance(column_data, np.ndarray):
            column_data = pd.Series(column_data, name=column_name)

        # If unit information is available, convert the unit
        if not isinstance(column_data, pd.Series):
            column_data = self._apply_unit_conversion(table_data=table_data, table=table_name, field=column_name)
            if not isinstance(column_data, pd.Series):
                raise TypeError(
                    f"The '{column_name}' column should now be unitless, "
                    f"but it still contains a unit: {column_data.columns.values}"
                )

        return self._apply_value_substitution(column_data=column_data, table=table_name, field=column_name)

    def _apply_value_substitution(self, column_data: pd.Series, table: str, field: str) -> pd.Series:

        if self._substitution is None:  # No substitution defined, at all
            return column_data

        # Find substitutions, ignore if none is found
        try:
            substitutions = self._substitution.get_substitutions(attr=field, table=table)
        except KeyError:
            return column_data

        if substitutions is None:  # No substitution defined, for this column
            return column_data

        def sub(value):
            try:
                return substitutions[value]
            except KeyError as ex:
                raise KeyError(
                    f"Unknown substitution for value '{value}' in column '{field}' in table '{table}'"
                ) from ex

        # Apply substitutions
        self._log.debug("Apply value substitutions", table=table, field=field, substitutions=substitutions)
        return column_data.map(sub)

    def _apply_unit_conversion(self, table_data: pd.DataFrame, table: str, field: str) -> pd.Series:
        unit = table_data[field].columns[0]

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
                table_data[field] *= multiplier
            except TypeError as ex:
                self._log.warning(
                    f"The column '{field}' on table '{table}' does not seem to be numerical "
                    f"while trying to apply a multiplier ({multiplier}) for unit '{unit}': {ex}"
                )

            # Replace the unit with the SI unit
            table_data.columns = table_data.columns.values
            table_data.columns = pd.MultiIndex.from_tuples(table_data.rename(columns={(field, unit): (field, si_unit)}))

        return table_data[pd.MultiIndex.from_tuples([(field, si_unit)])[0]]

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

    def items(self) -> Iterable[Tuple[str, Union[pd.DataFrame, np.ndarray]]]:
        """
        Mimic the dictionary .items() function
        """
        return self._data.items()
