# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Tabular Data Converter: Load data from multiple tables and use a mapping file to convert the data to PGM
"""

from pathlib import Path
from typing import Any, Dict, List, Literal, Mapping, Optional, Union, cast

import numpy as np
import pandas as pd
import yaml
from power_grid_model import initialize_array
from power_grid_model.data_types import Dataset

from power_grid_model_io.converters.base_converter import BaseConverter
from power_grid_model_io.data_stores.base_data_store import BaseDataStore
from power_grid_model_io.data_types import ExtraInfoLookup, TabularData
from power_grid_model_io.mappings.multiplier_mapping import MultiplierMapping, Multipliers
from power_grid_model_io.mappings.tabular_mapping import InstanceAttributes, Tables, TabularMapping
from power_grid_model_io.mappings.unit_mapping import UnitMapping, Units
from power_grid_model_io.mappings.value_mapping import ValueMapping, Values
from power_grid_model_io.utils.modules import get_function

MappingFile = Dict[Literal["multipliers", "grid", "units", "substitutions"], Union[Multipliers, Tables, Units, Values]]


class TabularConverter(BaseConverter[TabularData]):
    """Tabular Data Converter: Load data from multiple tables and use a mapping file to convert the data to PGM"""

    def __init__(
        self,
        mapping_file: Optional[Path] = None,
        source: Optional[BaseDataStore[TabularData]] = None,
        destination: Optional[BaseDataStore[TabularData]] = None,
    ):
        """
        Prepare some member variables and optionally load a mapping file

        Args:
            mapping_file: A yaml file containing the mapping.
        """
        super().__init__(source=source, destination=destination)
        self._mapping: TabularMapping = TabularMapping(mapping={})
        self._units: Optional[UnitMapping] = None
        self._substitutions: Optional[ValueMapping] = None
        self._multipliers: Optional[MultiplierMapping] = None
        if mapping_file is not None:
            self.set_mapping_file(mapping_file=mapping_file)

    def set_mapping_file(self, mapping_file: Path) -> None:
        """Read, parse and interpret a mapping file. This includes:
         * the table to table mapping ('grid')
         * the unit conversions ('units')
         * the value substitutions ('substitutions') (e.g. enums or other one-on-one value mapping)

        Args:
          mapping_file: Path:

        Returns:

        """
        # Read mapping
        if mapping_file.suffix.lower() != ".yaml":
            raise ValueError(f"Mapping file should be a .yaml file, {mapping_file.suffix} provided.")
        self._log.debug("Read mapping file", mapping_file=mapping_file)

        with open(mapping_file, "r", encoding="utf-8") as mapping_stream:
            mapping: MappingFile = yaml.safe_load(mapping_stream)
        if "grid" not in mapping:
            raise KeyError("Missing 'grid' mapping in mapping_file")
        self._mapping = TabularMapping(cast(Tables, mapping["grid"]))
        self._units = UnitMapping(cast(Units, mapping["units"])) if "units" in mapping else None
        self._substitutions = (
            ValueMapping(cast(Values, mapping["substitutions"])) if "substitutions" in mapping else None
        )
        self._multipliers = (
            MultiplierMapping(cast(Multipliers, mapping["multipliers"])) if "multipliers" in mapping else None
        )

    def _parse_data(self, data: TabularData, data_type: str, extra_info: Optional[ExtraInfoLookup]) -> Dataset:
        """This function parses tabular data and returns power-grid-model data

        Args:
          data: TabularData, i.e. a dictionary with the components as keys and pd.DataFrames as values, with
        attribute names as columns and their values in the table
          data_type: power-grid-model data type, i.e. "input" or "update"
          extra_info: an optional dictionary where extra component info (that can't be specified in
        power-grid-model data) can be specified
          data: TabularData:
          data_type: str:
          extra_info: Optional[ExtraInfoLookup]:

        Returns:
          a power-grid-model dataset, i.e. a dictionary as {component: np.ndarray}

        """
        # Apply units and substitutions to the data. Note that the conversions are 'lazy', i.e. the units and
        # substitutions will be applied the first time .get_column(table, field) is called.
        if self._units is not None:
            data.set_unit_multipliers(self._units)
        if self._substitutions is not None:
            data.set_substitutions(self._substitutions)

        # Initialize some empty data structures
        pgm: Dict[str, List[np.ndarray]] = {}

        # For each table in the mapping
        for table in self._mapping.tables():
            if table not in data or len(data[table]) == 0:
                continue  # pragma: no cover (bug in python 3.9)
            for component, attributes in self._mapping.instances(table=table):
                component_data = self._convert_table_to_component(
                    data=data,
                    data_type=data_type,
                    table=table,
                    component=component,
                    attributes=attributes,
                    extra_info=extra_info,
                )
                if component_data is not None:
                    if component not in pgm:
                        pgm[component] = []
                    pgm[component].append(component_data)

        input_data = TabularConverter._merge_pgm_data(data=pgm)
        self._log.debug(
            "Converted tabular data to power grid model data",
            n_components=len(input_data),
            n_instances=sum(len(table) for table in input_data.values()),
        )
        return input_data

    # pylint: disable = too-many-arguments
    def _convert_table_to_component(
        self,
        data: TabularData,
        data_type: str,
        table: str,
        component: str,
        attributes: InstanceAttributes,
        extra_info: Optional[ExtraInfoLookup],
    ) -> Optional[np.ndarray]:
        """
        This function converts a single table/sheet of TabularData to a power-grid-model input/update array. One table
        corresponds to one component

        Args:
          data: The full dataset with tabular data
          data_type: The data type, i.e. "input" or "update"
          table: The name of the table that should be converter
          component: the component for which a power-grid-model array should be made
          attributes: a dictionary with a mapping from the attribute names in the table to the corresponding
        power-grid-model attribute names
          extra_info: an optional dictionary where extra component info (that can't be specified in
        power-grid-model data) can be specified
          data: TabularData:
          data_type: str:
          table: str:
          component: str:
          attributes: InstanceAttributes:
          extra_info: Optional[ExtraInfoLookup]:

        Returns:
          returns a power-grid-model structured array for one component

        """
        if table not in data:
            return None

        n_records = len(data[table])

        try:
            pgm_data = initialize_array(data_type=data_type, component_type=component, shape=n_records)
        except KeyError as ex:
            raise KeyError(f"Invalid component type '{component}' or data type '{data_type}'") from ex

        if "id" not in attributes:
            raise KeyError(f"No mapping for the attribute 'id' for '{component}s'!")

        # Make sure that the "id" column is always parsed first (at least before "extra" is parsed)
        sorted_attributes = sorted(attributes.items(), key=lambda x: "" if x[0] == "id" else x[0])

        for attr, col_def in sorted_attributes:
            self._convert_col_def_to_attribute(
                data=data,
                pgm_data=pgm_data,
                table=table,
                component=component,
                attr=attr,
                col_def=col_def,
                extra_info=extra_info,
            )

        return pgm_data

    # pylint: disable = too-many-arguments
    def _convert_col_def_to_attribute(
        self,
        data: TabularData,
        pgm_data: np.ndarray,
        table: str,
        component: str,
        attr: str,
        col_def: Any,
        extra_info: Optional[ExtraInfoLookup],
    ):
        """This function updates one of the attributes of pgm_data, based on the corresponding table/column in a tabular
        dataset

        Args:
          data: TabularData, i.e. a dictionary with the components as keys and pd.DataFrames as values, with
        attribute names as columns and their values in the table
          pgm_data: a power-grid-model input/update array for one component
          table: the table name of the particular component in the tabular dataset
          component: the corresponding component
          attr: the name of the attribute that should be updated in the power-grid-model array
          col_def: the name of the column where the attribute values can be found
          extra_info: an optional dictionary where extra component info (that can't be specified in
        power-grid-model data) can be specified
          data: TabularData:
          pgm_data: np.ndarray:
          table: str:
          component: str:
          attr: str:
          col_def: Any:
          extra_info: Optional[ExtraInfoLookup]:

        Returns:
          the function updates pgm_data, it should not return something

        """
        # To avoid mistakes, the attributes in the mapping should exist. There is one extra attribute called
        # 'extra' in which extra information can be captured.
        if attr not in pgm_data.dtype.names and attr != "extra":
            attrs = ", ".join(pgm_data.dtype.names)
            raise KeyError(f"Could not find attribute '{attr}' for '{component}s'. (choose from: {attrs})")

        if attr == "extra":
            # Extra info must be linked to the object IDs, therefore the uuids should be known before extra info can
            # be parsed. Before this for loop, it is checked that "id" exists and it is placed at the front.
            self._handle_extra_info(
                data=data, table=table, col_def=col_def, uuids=pgm_data["id"], extra_info=extra_info
            )
            # Extra info should not be added to the numpy arrays, so let's continue to the next attribute
            return

        attr_data = self._parse_col_def(data=data, table=table, col_def=col_def, extra_info=extra_info)

        if len(attr_data.columns) != 1:
            raise ValueError(f"DataFrame for {component}.{attr} should contain a single column ({attr_data.columns})")

        pgm_data[attr] = attr_data.iloc[:, 0]

    def _handle_extra_info(
        self,
        data: TabularData,
        table: str,
        col_def: Any,
        uuids: np.ndarray,
        extra_info: Optional[ExtraInfoLookup],
    ) -> None:
        """This function can extract extra info from the tabular data and store it in the extra_info dict

        Args:
          data: tabularData, i.e. a dictionary with the components as keys and pd.DataFrames as values, with
        attribute names as columns and their values in the table
          table: the table name of the particular component in the tabular dataset
          col_def: the name of the column that should be stored in extra_info
          uuids: a numpy nd.array containing the uuids of the components
          extra_info: an optional dictionary where extra component info (that can't be specified in
        power-grid-model data) can be specified
          data: TabularData:
          table: str:
          col_def: Any:
          uuids: np.ndarray:
          extra_info: Optional[ExtraInfoLookup]:

        Returns:

        """
        if extra_info is None:
            return

        extra = self._parse_col_def(data=data, table=table, col_def=col_def, extra_info=None).to_dict(orient="records")
        for i, xtr in zip(uuids, extra):
            xtr = {k: v for k, v in xtr.items() if not isinstance(v, float) or not np.isnan(v)}
            if xtr:
                if i in extra_info:
                    extra_info[i].update(xtr)
                else:
                    extra_info[i] = xtr

    @staticmethod
    def _merge_pgm_data(data: Dict[str, List[np.ndarray]]) -> Dict[str, np.ndarray]:
        """During the conversion, multiple numpy arrays can be produced for the same type of component. These arrays
        should be concatenated to form one large table.

        Args:
          data: For each component, one or more numpy structured arrays
          data: Dict[str:
          List[np.ndarray]]:

        Returns:

        """
        merged = {}
        for component_name, data_set in data.items():

            # If there is only one array, use it as is
            if len(data_set) == 1:
                merged[component_name] = data_set[0]

            # If there are multiple arrays, concatenate them
            elif len(data_set) > 1:
                # pylint: disable=unexpected-keyword-arg
                merged[component_name] = np.concatenate(data_set, dtype=data_set[0].dtype)

        return merged

    def _serialize_data(self, data: Dataset, extra_info: Optional[ExtraInfoLookup]) -> TabularData:
        if extra_info is not None:
            raise NotImplementedError("Extra info can not (yet) be stored for tabular data")
        if isinstance(data, list):
            raise NotImplementedError("Batch data can not (yet) be stored for tabular data")
        return TabularData(**data)

    def _parse_col_def(
        self, data: TabularData, table: str, col_def: Any, extra_info: Optional[ExtraInfoLookup]
    ) -> pd.DataFrame:
        """Interpret the column definition and extract/convert/create the data as a pandas DataFrame.

        Args:
          data: TabularData:
          table: str:
          col_def: Any:
          extra_info: Optional[ExtraInfoLookup]:

        Returns:

        """
        if isinstance(col_def, (int, float)):
            return self._parse_col_def_const(data=data, table=table, col_def=col_def)
        if isinstance(col_def, str):
            return self._parse_col_def_column_name(data=data, table=table, col_def=col_def)
        if isinstance(col_def, dict):
            return self._parse_col_def_filter(data=data, table=table, col_def=col_def, extra_info=extra_info)
        if isinstance(col_def, list):
            return self._parse_col_def_composite(data=data, table=table, col_def=col_def)
        raise TypeError(f"Invalid column definition: {col_def}")

    @staticmethod
    def _parse_col_def_const(data: TabularData, table: str, col_def: Union[int, float]) -> pd.DataFrame:
        """Create a single column pandas DataFrame containing the const value.

        Args:
          data: TabularData:
          table: str:
          col_def: Union[int:
          float]:

        Returns:

        """
        assert isinstance(col_def, (int, float))
        return pd.DataFrame([col_def] * len(data[table]))

    def _parse_col_def_column_name(self, data: TabularData, table: str, col_def: str) -> pd.DataFrame:
        """Extract a column from the data. If the column doesn't exist, check if the col_def is a special float value,
        like 'inf'. If that's the case, create a single column pandas DataFrame containing the const value.

        Args:
          data: TabularData:
          table: str:
          col_def: str:

        Returns:

        """
        assert isinstance(col_def, str)
        table_data = data[table]

        # If multiple columns are given in col_def, return the first column that exists in the dataset
        columns = [col_name.strip() for col_name in col_def.split("|")]
        for col_name in columns:
            if col_name in table_data or col_name == "index":
                col_data = data.get_column(table_name=table, column_name=col_name)
                col_data = self._apply_multiplier(table=table, column=col_name, data=col_data)
                return pd.DataFrame(col_data)

        try:  # Maybe it is not a column name, but a float value like 'inf', let's try to convert the string to a float
            const_value = float(col_def)
        except ValueError:
            # pylint: disable=raise-missing-from
            columns_str = " and ".join(f"'{col_name}'" for col_name in columns)
            raise KeyError(f"Could not find column {columns_str} on table '{table}'")

        return self._parse_col_def_const(data=data, table=table, col_def=const_value)

    def _apply_multiplier(self, table: str, column: str, data: pd.Series) -> pd.Series:
        if self._multipliers is None:
            return data
        try:
            multiplier = self._multipliers.get_multiplier(table=table, attr=column)
            self._log.debug("Applied multiplier", table=table, column=column, multiplier=multiplier)
            return data * multiplier
        except KeyError:
            return data

    def _parse_reference(
        self, data: TabularData, table: str, other_table: str, query_column: str, key_column: str, value_column: str
    ) -> pd.DataFrame:
        """
        Find and extract a column from a different table.

        Args:
            data: The data
            table: The current table named
            other_table: The table in which we would like to find a value
            key_column: The column in the current table that stores the keys
            query_column: The column in the other table in which we should look for the keys
            value_column: The column in the other table which stores the values that we would like to return

        Returns:

        """
        queries = self._parse_col_def_column_name(data=data, table=table, col_def=query_column)
        keys = self._parse_col_def_column_name(data=data, table=other_table, col_def=key_column)
        values = self._parse_col_def_column_name(data=data, table=other_table, col_def=value_column)
        other = pd.concat([keys, values], axis=1)
        result = queries.merge(other, how="left", left_on=query_column, right_on=key_column)
        return result[[value_column]]

    def _parse_col_def_filter(
        self, data: TabularData, table: str, col_def: Dict[str, Any], extra_info: Optional[ExtraInfoLookup]
    ) -> pd.DataFrame:
        """
        Parse column filters like 'auto_id', 'reference', 'function', etc
        """
        assert isinstance(col_def, dict)
        data_frames = []
        for name, sub_def in col_def.items():
            if name == "auto_id":
                # Check that "key" is in the definition and no other keys than "table" and "name"
                if (
                    not isinstance(sub_def, dict)
                    or "key" not in sub_def
                    or len(set(sub_def.keys()) & {"table", "name"}) > 2
                ):
                    raise ValueError(f"Invalid {name} definition: {sub_def}")
                col_data = self._parse_auto_id(
                    data=data,
                    table=table,
                    ref_table=sub_def.get("table"),
                    ref_name=sub_def.get("name"),
                    key_col_def=sub_def["key"],
                    extra_info=extra_info,
                )
            elif name == "reference":
                # Check that (only) the required keys are in the definition
                if not isinstance(sub_def, dict) or {
                    "other_table",
                    "query_column",
                    "key_column",
                    "value_column",
                } != set(sub_def.keys()):
                    raise ValueError(f"Invalid {name} definition: {sub_def}")
                return self._parse_reference(
                    data=data,
                    table=table,
                    other_table=sub_def["other_table"],
                    query_column=sub_def["query_column"],
                    key_column=sub_def["key_column"],
                    value_column=sub_def["value_column"],
                )
            elif isinstance(sub_def, list):
                col_data = self._parse_pandas_function(data=data, table=table, function=name, col_def=sub_def)
            elif isinstance(sub_def, dict):
                col_data = self._parse_function(data=data, table=table, function=name, col_def=sub_def)
            else:
                raise TypeError(f"Invalid {name} definition: {sub_def}")
            data_frames.append(col_data)
        return pd.concat(data_frames, axis=1)

    def _parse_auto_id(
        self,
        data: TabularData,
        table: str,
        ref_table: Optional[str],
        ref_name: Optional[str],
        key_col_def: Union[str, List[str], Dict[str, str]],
        extra_info: Optional[ExtraInfoLookup],
    ) -> pd.DataFrame:
        """
        Create (or retrieve) a unique numerical id for each object (row) in `data[table]`, based on the `name`
        attribute, which is constant for each object, and the value(s) of `key_col_def`, which describes most likely a
        single column, or a list of columns.

        Args:
            data: The entire input data
            table: The current table name
            ref_table: The table name to which the id refers. If None, use the current table name.
            ref_name: A custom textual identifier, to be used for the auto_id. If None, ignore it.
            key_col_def: A column definition which should be unique for each object within the current table

        Returns: A single column containing numerical ids

        """

        # Handle reference table
        # mypy complains about ref_table being optional, therefore ref_table_str is defined as a string
        ref_table_str = ref_table or table

        # Handle reference column definition
        if isinstance(key_col_def, dict):
            key_names = list(key_col_def.keys())
            key_col_def = list(key_col_def.values())
        elif isinstance(key_col_def, list):
            key_names = key_col_def
        elif isinstance(key_col_def, str):
            key_names = [key_col_def]
        else:
            raise TypeError(f"Invalid key definition type '{type(key_col_def).__name__}': {key_col_def}")

        col_data = self._parse_col_def(data=data, table=table, col_def=key_col_def, extra_info=None)

        def auto_id(row: np.ndarray):
            key = dict(zip(key_names, row))
            pgm_id = self._get_id(table=ref_table_str, key=key, name=ref_name)

            # Extra info should only be added for the "id" field. Unfortunately we cannot check the field name at
            # this point, so we'll use a heuristic:
            # 1. An extra info dictionary should be supplied (i.e. extra info is requested by the caller).
            # 2. The auto_id should refer to the current table.
            #    (a counter example is an auto id referring to the nodes table, while the current table is lines)
            # 3. There shouldn't be any extra info for the current pgm_id, because the id attribute is supposed to be
            #    the first argument to be parsed.
            if extra_info is not None and ref_table_str == table and pgm_id not in extra_info:
                if ref_name is not None:
                    extra_info[pgm_id] = {"id_reference": {"table": ref_table_str, "name": ref_name, "key": key}}
                else:
                    extra_info[pgm_id] = {"id_reference": {"table": ref_table_str, "key": key}}

            return pgm_id

        return col_data.apply(auto_id, axis=1, raw=True)

    def _parse_pandas_function(self, data: TabularData, table: str, function: str, col_def: List[Any]) -> pd.DataFrame:
        """Special vectorized functions.

        Args:
          data: The data
          table: The name of the current table
          function: The name of the function.
          col_def: The definition of the function arguments

        Returns:

        """
        assert isinstance(col_def, list)

        # "multiply" is an alias for "prod"
        if function == "multiply":
            function = "prod"

        col_data = self._parse_col_def(data=data, table=table, col_def=col_def, extra_info=None)

        try:
            fn_ptr = getattr(col_data, function)
        except AttributeError as ex:
            raise ValueError(f"Pandas DataFrame has no function '{function}'") from ex

        try:
            return pd.DataFrame(fn_ptr(axis=1))
        except TypeError as ex:
            raise ValueError(f"Invalid pandas function DataFrame.{function}") from ex

    def _parse_function(self, data: TabularData, table: str, function: str, col_def: Dict[str, Any]) -> pd.DataFrame:
        """Import the function by name and apply it to each row.

        Args:
          data: The data
          table: The name of the current table
          function: The name (or path) of the function.
          col_def: The definition of the function keyword arguments

        Returns:

        """
        assert isinstance(col_def, dict)

        fn_ptr = get_function(function)
        key_words = list(col_def.keys())
        sub_def = list(col_def.values())
        col_data = self._parse_col_def(data=data, table=table, col_def=sub_def, extra_info=None)

        if col_data.empty:
            raise ValueError(f"Cannot apply function {function} to an empty DataFrame")

        col_data = col_data.apply(lambda row, fn=fn_ptr: fn(**dict(zip(key_words, row))), axis=1, raw=True)
        return pd.DataFrame(col_data)

    def _parse_col_def_composite(self, data: TabularData, table: str, col_def: list) -> pd.DataFrame:
        """Select multiple columns (each is created from a column definition) and return them as a new DataFrame.

        Args:
          data: TabularData:
          table: str:
          col_def: list:

        Returns:

        """
        assert isinstance(col_def, list)
        columns = [self._parse_col_def(data=data, table=table, col_def=sub_def, extra_info=None) for sub_def in col_def]
        return pd.concat(columns, axis=1)

    def _get_id(self, table: str, key: Mapping[str, int], name: Optional[str]) -> int:
        """
        Get a unique numerical ID for the supplied name / key combination
        Args:
            table: Table name (e.g. "Nodes")
            key: Component identifier (e.g. {"name": "node1"} or {"number": 1, "sub_number": 2})
            name: Optional component name (e.g. "internal_node")
        Returns: A unique id
        """
        auto_id_key = (table, tuple(sorted(key.items())), name)
        return self._auto_id(item=(table, key, name), key=auto_id_key)

    def get_id(self, table: str, key: Mapping[str, int], name: Optional[str] = None) -> int:
        """
        Get a the numerical ID previously associated with the supplied name / key combination
        Args:
            table: Table name (e.g. "Nodes")
            key: Component identifier (e.g. {"name": "node1"} or {"number": 1, "sub_number": 2})
            name: Optional component name (e.g. "internal_node")
        Returns: The associated id
        """
        auto_id_key = (table, tuple(sorted(key.items())), name)
        if auto_id_key not in self._auto_id:
            raise KeyError((table, key, name))
        return self._auto_id(item=(table, key, name), key=auto_id_key)

    def lookup_id(self, pgm_id: int) -> Dict[str, Union[str, Dict[str, int]]]:
        """
        Retrieve the original name / key combination of a pgm object
        Args:
            pgm_id: a unique numerical ID
        Returns: The original name / key combination
        """
        table, key, name = self._auto_id[pgm_id]
        reference = {"table": table}
        if name is not None:
            reference["name"] = name
        reference["key"] = key
        return reference
