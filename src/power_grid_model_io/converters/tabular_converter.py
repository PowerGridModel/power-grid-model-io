# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Tabular Data Converter: Load data from multiple tables and use a mapping file to convert the data to PGM
"""

import re
from importlib import import_module
from pathlib import Path
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union, cast

import numpy as np
import pandas as pd
from power_grid_model import initialize_array
from power_grid_model.data_types import Dataset

from power_grid_model_io.converters.base_converter import BaseConverter
from power_grid_model_io.data_types import ExtraInfoLookup, TabularData
from power_grid_model_io.mappings.tabular_mapping import InstanceAttributes, Tables, TabularMapping
from power_grid_model_io.mappings.unit_mapping import UnitMapping, Units
from power_grid_model_io.mappings.value_mapping import ValueMapping, Values
from power_grid_model_io.utils.auto_id import AutoID
from power_grid_model_io.utils.modules import import_optional_module

yaml = import_optional_module("tabular", "yaml")

"""
Regular expressions to match patterns like:
  OtherTable!ValueColumn[IdColumn=RefColumn]
and:
  OtherTable!ValueColumn[OtherTable!IdColumn=ThisTable!RefColumn]
  
([^!]+)     OtherTable
!           separator
([^\[]+)    ValueColumn
[           separator
([^\[]+)    ValueColumn
(([^!]+)!)? OtherTable + separator! (optional)
([^=]+)     IdColumn
=           separator
(([^!]+)!)? ThisTable + separator! (optional)
=           separator
([^\]]+)
]           separator
"""
COL_REF_RE = re.compile(r"([^!]+)!([^\[]+)\[(([^!]+)!)?([^=]+)=(([^!]+)!)?([^\]]+)\]")


class TabularConverter(BaseConverter[TabularData]):
    """
    Tabular Data Converter: Load data from multiple tables and use a mapping file to convert the data to PGM
    """

    def __init__(self, mapping_file: Optional[Path] = None):
        """
        Prepare some member variables and optionally load a mapping file

        Args:
            mapping_file: A yaml file containing the mapping.
        """
        super().__init__()
        self._mapping: TabularMapping = TabularMapping(mapping={})
        self._units: Optional[UnitMapping] = None
        self._substitution: Optional[ValueMapping] = None
        if mapping_file is not None:
            self.set_mapping_file(mapping_file=mapping_file)

    def set_mapping_file(self, mapping_file: Path) -> None:
        """
        Read, parse and interpret a mapping file. This includes:
         * the table to table mapping ('grid')
         * the unit conversions ('units')
         * the value substitutions ('substitutions') (e.g. enums or other one-on-one value mapping)
        """
        # Read mapping
        if mapping_file.suffix.lower() != ".yaml":
            raise ValueError(f"Mapping file should be a .yaml file, {mapping_file.suffix} provided.")
        self._log.debug("Read mapping file", mapping_file=mapping_file)

        with open(mapping_file, "r", encoding="utf-8") as mapping_stream:
            mapping: Dict[Literal["grid", "units", "substitutions"], Union[Tables, Units, Values]] = yaml.safe_load(
                mapping_stream
            )
        if "grid" not in mapping:
            raise KeyError("Missing 'grid' mapping in mapping_file")
        self._mapping = TabularMapping(cast(Tables, mapping["grid"]))
        self._units = UnitMapping(cast(Units, mapping["units"])) if "units" in mapping else None
        self._substitutions = (
            ValueMapping(cast(Values, mapping["substitutions"])) if "substitutions" in mapping else None
        )

    def _parse_data(self, data: TabularData, data_type: str, extra_info: Optional[ExtraInfoLookup] = None) -> Dataset:

        # Apply units and substitutions to the data. Note that the conversions are 'lazy', i.e. the units and
        # substitutions will be applied the first time .get_column(table, field) is called.
        data.set_units(self._units)
        data.set_substitutions(self._substitutions)

        # Initialize some empty data structures
        pgm: Dict[str, List[np.ndarray]] = {}
        extra: Dict[int, Dict[str, Any]] = {}
        lookup = AutoID()

        # For each table in the mapping
        for table in self._mapping.tables():
            for component, attributes in self._mapping.instances(table=table):
                component_data = self._convert_table_to_component(
                    data=data,
                    data_type=data_type,
                    table=table,
                    component=component,
                    attributes=attributes,
                    lookup=lookup,
                    extra_info=extra_info,
                )
                if component_data is not None:
                    if component not in pgm:
                        pgm[component] = []
                    pgm[component].append(component_data)

        input_data = self._merge_pgm_data(data=pgm, data_type=data_type)
        self._log.debug(
            "Converted tabular data to power grid model data",
            n_components=len(input_data),
            n_instances=sum(len(table) for table in input_data.values()),
        )
        return input_data

    def _convert_table_to_component(
        self,
        data: TabularData,
        data_type: str,
        table: str,
        component: str,
        attributes: InstanceAttributes,
        lookup: AutoID,
        extra_info: Optional[ExtraInfoLookup],
    ) -> Optional[np.ndarray]:
        if table not in data:
            return None, {}

        n_records = len(data[table])

        try:
            pgm_data = initialize_array(data_type=data_type, component_type=component, shape=n_records)
        except KeyError as ex:
            raise KeyError(f"Invalid component type '{component}'") from ex

        for attr, col_def in attributes.items():

            if attr not in pgm_data.dtype.names and attr != "extra":
                attrs = ", ".join(pgm_data.dtype.names)
                raise KeyError(f"Could not find attribute '{attr}' for '{component}'. (choose from: {attrs})")

            col_data = _parse_col_def(table=data, sheet_name=table, col_def=col_def)
            if attr == "extra":
                # Extra info is added when processing the id column
                continue
            if attr == "id":
                extra = col_data.to_dict(orient="records")
                col_data = col_data.apply(lambda row: _id_lookup(lookup, component, row), axis=1)
                if extra_info is not None:
                    for i, xtr in zip(col_data, extra):
                        extra_info[i] = {"table": table}
                        extra_info[i].update(xtr)
                    if "extra" in attributes:
                        extra = _parse_col_def(table=data, sheet_name=table, col_def=attributes["extra"])
                        if not extra.columns.is_unique:
                            extra = extra.loc[:, ~extra.columns.duplicated()]
                        extra = extra.to_dict(orient="records")
                        for i, xtr in zip(col_data, extra):
                            extra_info[i].update(
                                {k: v for k, v in xtr.items() if not isinstance(v, (int, float)) or not np.isnan(v)}
                            )
            elif attr.endswith("node"):
                col_data = col_data.apply(lambda row: _id_lookup(lookup, "node", row), axis=1)
            elif len(col_data.columns) != 1:
                raise ValueError(
                    f"DataFrame for {component}.{attr} should contain a single column " f"({col_data.columns})"
                )
            else:
                col_data = col_data.iloc[:, 0]
            try:
                pgm_data[attr] = col_data
            except ValueError as ex:
                if "invalid literal" in str(ex) and isinstance(col_def, str):
                    raise ValueError(
                        f"Possibly missing enum value for '{col_def}' column on '{table}' sheet: {ex}"
                    ) from ex
                raise

        return pgm_data

    def _merge_pgm_data(self, data: Dict[str, List[np.ndarray]], data_type: str) -> Dict[str, np.ndarray]:
        """
        During the conversion, multiple numpy arrays can be produced for the same type of componnent. These arrays
        should be concatenated to form one large table.

        Args:
            data: For each component, one or more numpy structured arrays
            data_type: The data_type defines the attributs in the numpy array (input, update, sym_output, asym_output).
        """
        merged = {}
        for component_name, data_set in data.items():

            # If there is only one array, use it as is
            if len(data_set) == 1:
                merged[component_name] = data_set[0]

            # If there are numtiple arrays, concatenate them
            elif len(data_set) > 1:
                merged[component_name] = np.concatenate(data_set)

        return merged

    def _serialize_data(self, data: Dataset, extra_info: Optional[ExtraInfoLookup] = None) -> TabularData:
        if extra_info is not None:
            raise NotImplementedError("Extra info can not (yet) be stored for tabular data")
        if isinstance(data, list):
            raise NotImplementedError("Batch data can not(yet) be stored for tabular data")
        return TabularData(**data)


def _parse_col_def(table: TabularData, sheet_name: str, col_def: Any) -> pd.DataFrame:
    """
    TODO: Revise this function and add it to TabularConverter
    """
    if isinstance(col_def, (int, float)):
        return _parse_col_def_const(workbook=table, sheet_name=sheet_name, col_def=col_def)
    if isinstance(col_def, str) and COL_REF_RE.fullmatch(col_def) is not None:
        return _parse_col_def_column_reference(workbook=table, sheet_name=sheet_name, col_def=col_def)
    if isinstance(col_def, str):
        return _parse_col_def_column_name(workbook=table, sheet_name=sheet_name, col_def=col_def)
    if isinstance(col_def, dict):
        return _parse_col_def_function(workbook=table, sheet_name=sheet_name, col_def=col_def)
    if isinstance(col_def, list):
        return _parse_col_def_composite(workbook=table, sheet_name=sheet_name, col_def=col_def)
    raise TypeError(f"Invalid column definition: {col_def}")


def _parse_col_def_const(workbook: TabularData, sheet_name: str, col_def: Union[int, float]) -> pd.DataFrame:
    """
    TODO: Revise this function and add it to TabularConverter
    """
    assert isinstance(col_def, (int, float))
    return pd.DataFrame([col_def] * len(workbook[sheet_name]))


def _parse_col_def_column_name(workbook: TabularData, sheet_name: str, col_def: str) -> pd.DataFrame:
    """
    TODO: Revise this function and add it to TabularConverter
    """
    assert isinstance(col_def, str)
    sheet = workbook[sheet_name]

    columns = [col_name.strip() for col_name in col_def.split("|")]
    for col_name in columns:
        if col_name in sheet:
            return pd.DataFrame(workbook.get_column(table=sheet_name, field=col_name))

    try:  # Maybe it is not a column name, but a float value like 'inf'
        const_value = float(col_def)
    except ValueError as ex:
        columns_str = " and ".join(f"'{col_name}'" for col_name in columns)
        raise KeyError(f"Could not find column {columns_str} on sheet '{sheet_name}'") from ex

    return _parse_col_def_const(workbook=workbook, sheet_name=sheet_name, col_def=const_value)


def _parse_col_def_column_reference(workbook: TabularData, sheet_name: str, col_def: str) -> pd.DataFrame:
    """
    TODO: Revise this function and add it to TabularConverter
    """
    assert isinstance(col_def, str)
    match = COL_REF_RE.fullmatch(col_def)
    if match is None:
        raise ValueError(
            f"Invalid column reference '{col_def}' " "(should be 'OtherSheet!ValueColumn[IdColumn=RefColumn])"
        )
    other_sheet, value_col_name, _, other_sheet_, id_col_name, _, this_sheet_, ref_col_name = match.groups()
    if (other_sheet_ is not None and other_sheet_ != other_sheet) or (
        this_sheet_ is not None and this_sheet_ != sheet_name
    ):
        raise ValueError(
            f"Invalid column reference '{col_def}'.\n"
            "It should be something like "
            f"{other_sheet}!{value_col_name}[{other_sheet}!{{id_column}}={sheet_name}!{{ref_column}}] "
            f"or simply {other_sheet}!{value_col_name}[{{id_column}}={{ref_column}}]"
        )
    ref_column = _parse_col_def_column_name(workbook=workbook, sheet_name=sheet_name, col_def=ref_col_name)
    id_column = _parse_col_def_column_name(workbook=workbook, sheet_name=other_sheet, col_def=id_col_name)
    val_column = _parse_col_def_column_name(workbook=workbook, sheet_name=other_sheet, col_def=value_col_name)
    other = pd.concat([id_column, val_column], axis=1)
    result = ref_column.merge(other, how="left", left_on=ref_col_name, right_on=id_col_name)
    return result[value_col_name]


def _parse_col_def_function(workbook: TabularData, sheet_name: str, col_def: Dict[str, str]) -> pd.DataFrame:
    """
    TODO: Revise this function and add it to TabularConverter
    """
    assert isinstance(col_def, dict)
    data = []
    for fn_name, sub_def in col_def.items():
        function = _get_function(fn_name)
        col_data = _parse_col_def(table=workbook, sheet_name=sheet_name, col_def=sub_def)
        col_data = col_data.apply(lambda row: function(*row), axis=1, raw=True)  # pylint: disable=cell-var-from-loop
        data.append(col_data)
    return pd.concat(data, axis=1)


def _parse_col_def_composite(workbook: TabularData, sheet_name: str, col_def: list) -> pd.DataFrame:
    """
    TODO: Revise this function and add it to TabularConverter
    """
    assert isinstance(col_def, list)
    columns = [_parse_col_def(table=workbook, sheet_name=sheet_name, col_def=sub_def) for sub_def in col_def]
    return pd.concat(columns, axis=1)


def _get_function(fn_name: str) -> Callable:
    """
    TODO: Revise this function and add it to TabularConverter
    """
    parts = fn_name.split(".")
    function_name = parts.pop()
    module_path = ".".join(parts) if parts else "builtins"
    try:
        module = import_module(module_path)
    except ModuleNotFoundError as ex:
        raise AttributeError(f"Function: {fn_name} does not exist") from ex
    try:
        function = getattr(module, function_name)
    except AttributeError as ex:
        raise AttributeError(f"Function: {function_name} does not exist in {module_path}") from ex
    return function


def _id_lookup(lookup: AutoID, component: str, row: pd.Series) -> int:
    """
    TODO: Revise this function and add it to TabularConverter
    """
    data = {col.split(".").pop(): val for col, val in sorted(row.to_dict().items(), key=lambda x: x[0])}
    key = component + ":" + ",".join(f"{k}={v}" for k, v in data.items())
    return lookup(item={"component": component, "row": data}, key=key)
