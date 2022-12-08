# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
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
        """
        Tabular data can either be a collection of pandas DataFrames and/or numpy structured arrays.
        The key word arguments will define the keys of the data.

        tabular_data = TabularData(foo=foo_data)
        tabular_data["foo"] --> foo_data

        Args:
            **tables: A collection of pandas DataFrames and/or numpy structured arrays
        """
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

    def set_unit_multipliers(self, units: UnitMapping) -> None:
        """
        Define unit multipliers.

        Args:
            units: A UnitMapping object defining all the units and their conversions (e.g. 1 MW = 1_000_000 W)
        """
        self._units = units

    def set_substitutions(self, substitution: ValueMapping) -> None:
        """
        Define value substitutions

        Args:
            substitution: A ValueMapping defining all value substitutions (e.g. "yes" -> 1)
        """
        self._substitution = substitution

    def get_column(self, table_name: str, column_name: str) -> pd.Series:
        """
        Select a column from a table, while applying unit conversions and value substitutions

        Args:
            table_name: The name of the table as supplied in the constructor
            column_name: The name of the column or "index" to get the index

        Returns:
            The required column, with unit conversions and value substitutions applied
        """
        table_data = self._data[table_name]

        # If the index 'column' is requested, but no column called 'index' exist,
        # return the index of the dataframe as if it were an actual column.
        if column_name == "index" and "index" not in table_data and hasattr(table_data, "index"):
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

        try:
            if self._units is None:
                raise KeyError(unit)
            multiplier, si_unit = self._units.get_unit_multiplier(unit)
        except KeyError as ex:
            raise KeyError(f"Unknown unit '{unit}' for column '{field}' in table '{table}'") from ex

        if unit == si_unit:
            self._log.debug("No unit conversion needed", table=table, field=field, unit=unit)
        else:
            self._log.debug(
                "Apply unit conversion", table=table, field=field, unit=unit, si_unit=si_unit, multiplier=multiplier
            )
            try:
                table_data[field] *= multiplier
            except TypeError as ex:
                raise TypeError(
                    f"The column '{field}' on table '{table}' (or the multiplier) does not seem to be numerical "
                    f"while trying to apply a multiplier ({multiplier}) for unit '{unit}': {ex}"
                ) from ex

            # Replace the unit with the SI unit
            table_data.columns = table_data.columns.values
            table_data.columns = pd.MultiIndex.from_tuples(table_data.rename(columns={(field, unit): (field, si_unit)}))

        return table_data[pd.MultiIndex.from_tuples([(field, si_unit)])[0]]

    def __contains__(self, table_name: str) -> bool:
        """
        Mimic the dictionary 'in' operator

        Args:
            table_name: The name of the table as supplied in the constructor.

        Returns: True if the table name was supplied in the constructor.
        """
        return table_name in self._data

    def __getitem__(self, table_name: str) -> Union[pd.DataFrame, np.ndarray]:
        """
        Mimic the dictionary [] operator. It returns the 'raw' table data as stored in memory. This can be either a
        pandas DataFrame or a numpy structured array. It is possible that some unit conversions have been applied by
        previous calls to get_column().

        Args:
            table_name: The name of the table as supplied in the constructor

        Returns: The 'raw' table data
        """
        return self._data[table_name]

    def keys(self) -> Iterable[str]:
        """
        Mimic the dictionary .keys() function

        Returns: An iterator over all table names as supplied in the constructor.
        """

        return self._data.keys()

    def items(self) -> Iterable[Tuple[str, Union[pd.DataFrame, np.ndarray]]]:
        """
        Mimic the dictionary .items() function

        Returns: An iterator over the table names and the raw table data
        """

        # Note: PyCharm complains about the type, but it is correct, as an ItemsView extends from
        # AbstractSet[Tuple[_KT_co, _VT_co]], which actually is compatible with Iterable[_KT_co, _VT_co]
        return self._data.items()
