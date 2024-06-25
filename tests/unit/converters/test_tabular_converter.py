# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from typing import Callable, Tuple
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from power_grid_model import initialize_array, power_grid_meta_data
from power_grid_model.data_types import SingleDataset

from power_grid_model_io.converters.tabular_converter import TabularConverter
from power_grid_model_io.data_types import ExtraInfo, TabularData
from power_grid_model_io.mappings.tabular_mapping import InstanceAttributes
from power_grid_model_io.mappings.unit_mapping import UnitMapping

MAPPING_FILE = Path(__file__).parents[2] / "data" / "config" / "mapping.yaml"


@pytest.fixture
def converter():
    converter = TabularConverter(mapping_file=MAPPING_FILE)
    return converter


@pytest.fixture
def pgm_node_empty():
    node = initialize_array("input", "node", 2)
    return {"node": node}


@pytest.fixture
def pgm_line_empty():
    line = initialize_array("input", "line", 2)
    return {"line": line}


@pytest.fixture
def pgm_power_sensor_empty():
    power_sensor = initialize_array("input", "sym_power_sensor", 1)
    return {"power_sensor": power_sensor}


@pytest.fixture
def tabular_data():
    nodes = pd.DataFrame(
        data=[[1, 10.5], [2, 0.4]], columns=pd.MultiIndex.from_tuples([("id_number", ""), ("u_nom", "kV")])
    )
    lines = pd.DataFrame(data=[[1, 100], [2, 200]], columns=["id_number", "from_node_side"])
    loads = pd.DataFrame(data=[[1, 1, 1], [2, 2, 0]], columns=["id_number", "node_id", "switching_status"])
    tabular_data = TabularData(nodes=nodes, lines=lines, loads=loads)
    return tabular_data


@pytest.fixture
def tabular_data_no_units_no_substitutions():
    nodes = pd.DataFrame(data=[[1, 10.5e3], [2, 400.0]], columns=["id_number", "u_nom"])
    lines = pd.DataFrame(data=[[1, 2], [3, 1]], columns=["id_number", "from_node_side"])
    tabular_data = TabularData(nodes=nodes, lines=lines)
    return tabular_data


def test_set_mapping_file(converter: TabularConverter):
    with pytest.raises(ValueError, match="Mapping file should be a .yaml file, .txt provided."):
        converter.set_mapping_file(mapping_file=Path("dummy/path.txt"))

    dummy_path = Path(__file__).parents[2] / "data" / "config" / "dummy_mapping.yaml"
    with pytest.raises(KeyError, match="Missing 'grid' mapping in mapping_file"):
        converter.set_mapping_file(mapping_file=dummy_path)

    converter.set_mapping_file(mapping_file=MAPPING_FILE)


def test_parse_data(converter: TabularConverter, tabular_data: TabularData):
    data = MagicMock()
    converter._parse_data(data=data, data_type="dummy", extra_info=None)
    data.set_unit_multipliers.assert_called_once()
    data.set_substitutions.assert_called_once()

    pgm_input_data = converter._parse_data(data=tabular_data, data_type="input", extra_info=None)
    assert list(pgm_input_data.keys()) == ["node", "line", "sym_load"]
    assert len(pgm_input_data["node"]) == 2
    assert (pgm_input_data["node"]["id"] == [0, 1]).all()
    assert (pgm_input_data["node"]["u_rated"] == [10.5e3, 400]).all()

    assert len(pgm_input_data["line"]) == 2
    assert (pgm_input_data["line"]["id"] == [2, 3]).all()
    assert (pgm_input_data["line"]["from_node"] == [0, 1]).all()

    assert len(pgm_input_data["sym_load"]) == 4
    assert (pgm_input_data["sym_load"]["id"] == [4, 5, 6, 7]).all()
    assert (pgm_input_data["sym_load"]["node"] == [0, 1, 0, 1]).all()
    assert (pgm_input_data["sym_load"]["status"] == [1, 0, 1, 0]).all()
    assert pgm_input_data["sym_load"].dtype == power_grid_meta_data["input"]["sym_load"].dtype


def test_convert_table_to_component(converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData):
    # if table does not exist in data _convert_table_to_component should return None
    none_data = converter._convert_table_to_component(
        data=tabular_data_no_units_no_substitutions,
        data_type="input",
        table="some_random_table",
        component="node",
        attributes={"key": "value"},
        extra_info=None,
    )
    assert none_data is None
    # wrong component
    with pytest.raises(KeyError, match="Invalid component type 'dummy' or data type 'input'"):
        converter._convert_table_to_component(
            data=tabular_data_no_units_no_substitutions,
            data_type="input",
            table="nodes",
            component="dummy",
            attributes={"key": "value"},
            extra_info=None,
        )
    # wrong data_type
    with pytest.raises(KeyError, match="Invalid component type 'node' or data type 'some_type'"):
        converter._convert_table_to_component(
            data=tabular_data_no_units_no_substitutions,
            data_type="some_type",
            table="nodes",
            component="node",
            attributes={"key": "value"},
            extra_info=None,
        )
    # no 'id' in attributes
    with pytest.raises(KeyError, match="No mapping for the attribute 'id' for 'nodes'!"):
        converter._convert_table_to_component(
            data=tabular_data_no_units_no_substitutions,
            data_type="input",
            table="nodes",
            component="node",
            attributes={"key": "value"},
            extra_info=None,
        )

    node_attributes: InstanceAttributes = {"id": "id_number", "u_rated": "u_nom"}
    pgm_node_data = converter._convert_table_to_component(
        data=tabular_data_no_units_no_substitutions,
        data_type="input",
        table="nodes",
        component="node",
        attributes=node_attributes,
        extra_info=None,
    )
    assert pgm_node_data is not None
    assert len(pgm_node_data) == 2
    assert (pgm_node_data["id"] == [1, 2]).all()
    assert (pgm_node_data["u_rated"] == [10.5e3, 400]).all()


def test_convert_table_to_component__filters(
    converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData
):
    converter._convert_col_def_to_attribute = MagicMock()
    converter._parse_table_filters = MagicMock()
    node_attributes_with_filter = {"id": "id_number", "u_rated": "u_nom", "filters": [{"test_fn": {}}]}
    converter._convert_table_to_component(
        data=tabular_data_no_units_no_substitutions,
        data_type="input",
        table="nodes",
        component="node",
        attributes=node_attributes_with_filter,
        extra_info=None,
    )
    converter._parse_table_filters.assert_called_once_with(
        data=tabular_data_no_units_no_substitutions,
        table="nodes",
        filtering_functions=node_attributes_with_filter["filters"],
    )


def test_convert_col_def_to_attribute(
    converter: TabularConverter,
    tabular_data_no_units_no_substitutions: TabularData,
    pgm_node_empty: SingleDataset,
):
    with pytest.raises(
        KeyError, match=r"Could not find attribute 'incorrect_attribute' for 'nodes'. " r"\(choose from: id, u_rated\)"
    ):
        converter._convert_col_def_to_attribute(
            data=tabular_data_no_units_no_substitutions,
            pgm_data=pgm_node_empty["node"],
            table="nodes",
            component="node",
            attr="incorrect_attribute",
            col_def="id_number",
            table_mask=None,
            extra_info=None,
        )

    # test extra info
    converter._convert_col_def_to_attribute(
        data=tabular_data_no_units_no_substitutions,
        pgm_data=pgm_node_empty["node"],
        table="nodes",
        component="node",
        attr="extra",
        col_def="u_nom",
        table_mask=None,
        extra_info={0: {}, 1: {}},
    )

    # test other attr
    converter._convert_col_def_to_attribute(
        data=tabular_data_no_units_no_substitutions,
        pgm_data=pgm_node_empty["node"],
        table="nodes",
        component="node",
        attr="u_rated",
        col_def="u_nom",
        table_mask=None,
        extra_info=None,
    )
    assert len(pgm_node_empty) == 1
    assert (pgm_node_empty["node"]["u_rated"] == [10500.0, 400.0]).all()

    with pytest.raises(
        ValueError,
        match=r"DataFrame for node.u_rated should contain a single column "
        r"\(Index\(\['id_number', 'u_nom'\], dtype='object'\)\)",
    ):
        converter._convert_col_def_to_attribute(
            data=tabular_data_no_units_no_substitutions,
            pgm_data=pgm_node_empty["node"],
            table="nodes",
            component="node",
            attr="u_rated",
            col_def=["id_number", "u_nom"],
            table_mask=None,
            extra_info=None,
        )


def test_handle_extra_info(converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData):
    uuids = np.array([0, 1])
    # possible to call function with extra_info = None
    converter._handle_extra_info(
        data=tabular_data_no_units_no_substitutions,
        table="nodes",
        col_def="u_nom",
        uuids=uuids,
        table_mask=None,
        extra_info=None,
    )
    # _handle_extra_info creates extra info entry for id's that don't exist and updates existing entries
    extra_info: ExtraInfo = {0: {"some_value": "some_key"}}
    converter._handle_extra_info(
        data=tabular_data_no_units_no_substitutions,
        table="nodes",
        col_def="u_nom",
        uuids=uuids,
        table_mask=None,
        extra_info=extra_info,
    )
    assert extra_info == {
        0: {"some_value": "some_key", "u_nom": 10500.0},
        1: {"u_nom": 400.0},
    }


def test_handle_extra_info__units(converter: TabularConverter, tabular_data: TabularData):
    # Arrange
    uuids = np.array([0, 1])
    extra_info: ExtraInfo = {}
    tabular_data._units = UnitMapping({"V": {"kV": 1000.0}})

    # Act
    converter._handle_extra_info(
        data=tabular_data,
        table="nodes",
        col_def="u_nom",
        uuids=uuids,
        table_mask=None,
        extra_info=extra_info,
    )

    # Assert
    assert extra_info == {0: {"u_nom": 10500.0}, 1: {"u_nom": 400.0}}


def test_merge_pgm_data(converter: TabularConverter):
    nodes_1 = initialize_array("input", "node", 2)
    nodes_1["id"] = [0, 1]

    nodes_2 = initialize_array("input", "node", 3)
    nodes_2["id"] = [2, 3, 4]
    data = {"node": [nodes_1, nodes_2]}

    merged = converter._merge_pgm_data(data)
    assert len(merged) == 1
    assert (merged["node"]["id"] == np.array([0, 1, 2, 3, 4])).all()


def test_serialize_data(converter: TabularConverter, pgm_node_empty: SingleDataset):
    with pytest.raises(NotImplementedError, match=r"Extra info can not \(yet\) be stored for tabular data"):
        converter._serialize_data(data=pgm_node_empty, extra_info={})
    with pytest.raises(NotImplementedError, match=r"Batch data can not \(yet\) be stored for tabular data"):
        converter._serialize_data(data=[], extra_info=None)  # type: ignore

    pgm_node_empty["node"]["id"] = [1, 2]
    pgm_node_empty["node"]["u_rated"] = [3.0, 4.0]
    tabular_data = converter._serialize_data(data=pgm_node_empty, extra_info=None)
    assert len(tabular_data) == 1
    assert (tabular_data["node"]["id"] == np.array([1, 2])).all()
    assert (tabular_data["node"]["u_rated"] == np.array([3.0, 4.0])).all()


def test_parse_col_def(converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData):
    with pytest.raises(TypeError, match=r"Invalid column definition: \(\)"):
        converter._parse_col_def(
            data=tabular_data_no_units_no_substitutions,
            table="table_name",
            col_def=(),
            table_mask=None,
            extra_info=None,
        )

    # type(col_def) == int
    with patch(
        "power_grid_model_io.converters.tabular_converter.TabularConverter._parse_col_def_const"
    ) as mock_parse_col_def_const:
        converter._parse_col_def(
            data=tabular_data_no_units_no_substitutions,
            table="nodes",
            col_def=50,
            table_mask=None,
            extra_info=None,
        )
        mock_parse_col_def_const.assert_called_once_with(
            data=tabular_data_no_units_no_substitutions, table="nodes", table_mask=None, col_def=50
        )

    # type(col_def) == float
    with patch(
        "power_grid_model_io.converters.tabular_converter.TabularConverter._parse_col_def_const"
    ) as mock_parse_col_def_const:
        converter._parse_col_def(
            data=tabular_data_no_units_no_substitutions,
            table="nodes",
            col_def=4.0,
            table_mask=None,
            extra_info=None,
        )
        mock_parse_col_def_const.assert_called_once_with(
            data=tabular_data_no_units_no_substitutions, table="nodes", col_def=4.0, table_mask=None
        )

    # type(col_def) == str (regular expression)
    with patch(
        "power_grid_model_io.converters.tabular_converter.TabularConverter._parse_reference"
    ) as mock_parse_reference:
        converter._parse_col_def(
            data=tabular_data_no_units_no_substitutions,
            table="lines",
            col_def={
                "reference": {
                    "query_column": "from_node_side",
                    "other_table": "nodes",
                    "key_column": "id_number",
                    "value_column": "u_nom",
                }
            },
            table_mask=None,
            extra_info=None,
        )
        mock_parse_reference.assert_called_once_with(
            data=tabular_data_no_units_no_substitutions,
            table="lines",
            other_table="nodes",
            query_column="from_node_side",
            key_column="id_number",
            value_column="u_nom",
            table_mask=None,
        )

    # type(col_def) == str
    with patch(
        "power_grid_model_io.converters.tabular_converter.TabularConverter._parse_col_def_column_name"
    ) as mock_parse_col_def_column_name:
        converter._parse_col_def(
            data=tabular_data_no_units_no_substitutions,
            table="nodes",
            col_def="col_name",
            table_mask=None,
            extra_info=None,
        )
        mock_parse_col_def_column_name.assert_called_once_with(
            data=tabular_data_no_units_no_substitutions, table="nodes", col_def="col_name", table_mask=None
        )

    # type(col_def) == dict
    with patch(
        "power_grid_model_io.converters.tabular_converter.TabularConverter._parse_col_def_filter"
    ) as mock_parse_col_def_filter:
        converter._parse_col_def(
            data=tabular_data_no_units_no_substitutions,
            table="nodes",
            col_def={},
            table_mask=None,
            extra_info=None,
        )
        mock_parse_col_def_filter.assert_called_once_with(
            data=tabular_data_no_units_no_substitutions,
            table="nodes",
            col_def={},
            table_mask=None,
            extra_info=None,
        )

    # type(col_def) == list
    with patch(
        "power_grid_model_io.converters.tabular_converter.TabularConverter._parse_col_def_composite"
    ) as mock_parse_col_def_composite:
        converter._parse_col_def(
            data=tabular_data_no_units_no_substitutions,
            table="nodes",
            col_def=[],
            table_mask=None,
            extra_info=None,
        )
        mock_parse_col_def_composite.assert_called_once_with(
            data=tabular_data_no_units_no_substitutions, table="nodes", col_def=[], table_mask=None
        )


def test_parse_col_def_const(converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData):
    with pytest.raises(AssertionError):
        converter._parse_col_def_const(
            data=tabular_data_no_units_no_substitutions, table="nodes", col_def="str", table_mask=None  # type: ignore
        )

    # type(col_def) == int
    col_int = converter._parse_col_def_const(
        data=tabular_data_no_units_no_substitutions, table="nodes", col_def=50, table_mask=None
    )
    assert_frame_equal(col_int, pd.DataFrame([50, 50]))

    # type(col_def) == float
    col_int = converter._parse_col_def_const(
        data=tabular_data_no_units_no_substitutions, table="nodes", col_def=3.0, table_mask=None
    )
    assert_frame_equal(col_int, pd.DataFrame([3.0, 3.0]))


def test_parse_col_def_const__no_filter(
    converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData
):
    col_int = converter._parse_col_def_const(
        data=tabular_data_no_units_no_substitutions, table="nodes", col_def=3.0, table_mask=None
    )
    assert_frame_equal(col_int, pd.DataFrame([3.0, 3.0]))


def test_parse_col_def_column_name(converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData):
    with pytest.raises(AssertionError):
        converter._parse_col_def_column_name(
            data=tabular_data_no_units_no_substitutions, table="nodes", col_def=1, table_mask=None  # type: ignore
        )

    df_multiple_columns = converter._parse_col_def_column_name(
        data=tabular_data_no_units_no_substitutions,
        table="nodes",
        col_def="  wrong_column  | id_number  | u_nom  ",
        table_mask=None,
    )
    assert_frame_equal(df_multiple_columns, pd.DataFrame([1, 2], columns=["id_number"]))

    df_inf = converter._parse_col_def_column_name(
        data=tabular_data_no_units_no_substitutions, table="nodes", col_def="inf", table_mask=None
    )
    assert_frame_equal(df_inf, pd.DataFrame([np.inf, np.inf]))

    with pytest.raises(KeyError, match="Could not find column 'a' and 'b' and 'c' on table 'nodes'"):
        converter._parse_col_def_column_name(
            data=tabular_data_no_units_no_substitutions,
            table="nodes",
            col_def="  a  | b  | c  ",
            table_mask=None,
        )


def test_parse_reference(converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData):
    # get lines.from_nodes where line id == node id
    df_lines_from_node_long = converter._parse_reference(
        data=tabular_data_no_units_no_substitutions,
        table="lines",
        other_table="nodes",
        query_column="from_node_side",
        key_column="id_number",
        value_column="u_nom",
        table_mask=None,
    )
    assert_frame_equal(df_lines_from_node_long, pd.DataFrame([400.0, 10.5e3], columns=["u_nom"]))


def test_parse_col_def_filter(converter: TabularConverter):
    # Act/Assert:
    with pytest.raises(AssertionError):
        converter._parse_col_def_filter(
            data=tabular_data_no_units_no_substitutions, table="", col_def=[], table_mask=None, extra_info=None  # type: ignore
        )

    with pytest.raises(TypeError, match="Invalid foo definition: 123"):
        converter._parse_col_def_filter(
            data=MagicMock(), table="", col_def={"foo": 123}, table_mask=None, extra_info=None
        )


@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._parse_function")
def test_parse_col_def_filter__function(mock_parse_function: MagicMock, converter: TabularConverter):
    # Arrange
    data = MagicMock()
    function_result = pd.DataFrame([1, 2])
    mock_parse_function.return_value = function_result

    # Act
    result = converter._parse_col_def_filter(
        data=data,
        table="nodes",
        col_def={"path.to.function": {"foo": "id_number", "bar": "u_nom"}},
        table_mask=None,
        extra_info=None,
    )

    # Assert
    mock_parse_function.assert_called_once_with(
        data=data,
        table="nodes",
        function="path.to.function",
        col_def={"foo": "id_number", "bar": "u_nom"},
        table_mask=None,
    )
    pd.testing.assert_frame_equal(result, function_result)


@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._parse_pandas_function")
def test_parse_col_def_filter__pandas_function(mock_parse_function: MagicMock, converter: TabularConverter):
    # Arrange
    data = MagicMock()
    function_result = pd.DataFrame([1, 2])
    mock_parse_function.return_value = function_result

    # Act
    result = converter._parse_col_def_filter(
        data=data, table="nodes", col_def={"multiply": ["id_number", "u_nom"]}, table_mask=None, extra_info=None
    )

    # Assert
    mock_parse_function.assert_called_once_with(
        data=data,
        table="nodes",
        fn_name="multiply",
        col_def=["id_number", "u_nom"],
        table_mask=None,
    )
    pd.testing.assert_frame_equal(result, function_result)


@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._parse_auto_id")
def test_parse_col_def_filter__auto_id(mock_parse_auto_id: MagicMock, converter: TabularConverter):
    # Arrange
    data = MagicMock()
    auto_id_result = pd.DataFrame([1, 2])
    extra_info = MagicMock()
    mock_parse_auto_id.return_value = auto_id_result

    # Act
    result = converter._parse_col_def_filter(
        data=data,
        table="lines",
        col_def={"auto_id": {"table": "nodes", "name": "dummy", "key": "from_node_side"}},
        table_mask=None,
        extra_info=extra_info,
    )

    # Assert
    mock_parse_auto_id.assert_called_once_with(
        data=data,
        table="lines",
        ref_table="nodes",
        ref_name="dummy",
        key_col_def="from_node_side",
        table_mask=None,
        extra_info=extra_info,
    )
    pd.testing.assert_frame_equal(result, auto_id_result)

    # Act/Assert:
    with pytest.raises(ValueError, match="Invalid auto_id definition: {'a': 1, 'b': 2}"):
        converter._parse_col_def_filter(
            data=data, table="", col_def={"auto_id": {"a": 1, "b": 2}}, table_mask=None, extra_info=None
        )


@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._parse_reference")
def test_parse_col_def_filter__reference(mock_parse_reference: MagicMock, converter: TabularConverter):
    # Arrange
    data = MagicMock()
    reference_result = MagicMock()
    mock_parse_reference.return_value = reference_result

    # Act
    result = converter._parse_col_def_filter(
        data=data,
        table="lines",
        col_def={
            "reference": {
                "query_column": "from_node_side",
                "other_table": "nodes",
                "key_column": "id_number",
                "value_column": "u_nom",
            }
        },
        table_mask=None,
        extra_info=None,
    )

    # Assert
    mock_parse_reference.assert_called_once_with(
        data=data,
        table="lines",
        other_table="nodes",
        query_column="from_node_side",
        key_column="id_number",
        value_column="u_nom",
        table_mask=None,
    )
    assert result is reference_result

    # Act/Assert:
    with pytest.raises(ValueError, match="Invalid reference definition: {'a': 1, 'b': 2}"):
        converter._parse_col_def_filter(
            data=data, table="", col_def={"reference": {"a": 1, "b": 2}}, table_mask=None, extra_info=None
        )


@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._get_id")
def test_parse_auto_id(
    mock_get_id: MagicMock, converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData
):
    # ref_table: None, ref_name: None, key_col_def: str, extra_info: None
    mock_get_id.side_effect = [101, 102]
    converter._parse_auto_id(
        data=tabular_data_no_units_no_substitutions,
        table="nodes",
        ref_table=None,
        ref_name=None,
        key_col_def="id_number",
        table_mask=None,
        extra_info=None,
    )
    mock_get_id.assert_has_calls(
        [call(table="nodes", key={"id_number": 1}, name=None), call(table="nodes", key={"id_number": 2}, name=None)]
    )


@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._get_id")
def test_parse_auto_id__extra_info(
    mock_get_id: MagicMock, converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData
):
    # ref_table: None, ref_name: None, key_col_def: str, extra_info: dict
    mock_get_id.side_effect = [101, 102]
    extra_info: ExtraInfo = {}
    converter._parse_auto_id(
        data=tabular_data_no_units_no_substitutions,
        table="nodes",
        ref_table=None,
        ref_name=None,
        key_col_def="id_number",
        table_mask=None,
        extra_info=extra_info,
    )
    mock_get_id.assert_has_calls(
        [call(table="nodes", key={"id_number": 1}, name=None), call(table="nodes", key={"id_number": 2}, name=None)]
    )
    assert extra_info[101] == {"id_reference": {"table": "nodes", "key": {"id_number": 1}}}
    assert extra_info[102] == {"id_reference": {"table": "nodes", "key": {"id_number": 2}}}


@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._get_id")
def test_parse_auto_id__reference_column(
    mock_get_id: MagicMock, converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData
):
    # ref_table: str, ref_name: None, key_col_def: dict, extra_info: dict
    mock_get_id.side_effect = [101, 102]
    extra_info: ExtraInfo = {}
    converter._parse_auto_id(
        data=tabular_data_no_units_no_substitutions,
        table="lines",
        ref_table="nodes",
        ref_name=None,
        key_col_def={"id_number": "from_node_side"},
        table_mask=None,
        extra_info=extra_info,
    )
    mock_get_id.assert_has_calls(
        [call(table="nodes", key={"id_number": 2}, name=None), call(table="nodes", key={"id_number": 1}, name=None)]
    )
    assert len(extra_info) == 0


@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._get_id")
def test_parse_auto_id__composite_key(
    mock_get_id: MagicMock, converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData
):
    # ref_table: None, ref_name: None, key_col_def: list, extra_info: dict
    mock_get_id.side_effect = [101, 102]
    extra_info: ExtraInfo = {}
    converter._parse_auto_id(
        data=tabular_data_no_units_no_substitutions,
        table="nodes",
        ref_table=None,
        ref_name=None,
        key_col_def=["id_number", "u_nom"],
        table_mask=None,
        extra_info=extra_info,
    )
    mock_get_id.assert_has_calls(
        [
            call(table="nodes", key={"id_number": 1, "u_nom": 10.5e3}, name=None),
            call(table="nodes", key={"id_number": 2, "u_nom": 400.0}, name=None),
        ]
    )
    assert extra_info[101] == {"id_reference": {"table": "nodes", "key": {"id_number": 1, "u_nom": 10.5e3}}}
    assert extra_info[102] == {"id_reference": {"table": "nodes", "key": {"id_number": 2, "u_nom": 400.0}}}


@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._get_id")
def test_parse_auto_id__named_objects(
    mock_get_id: MagicMock, converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData
):
    # ref_table: None, ref_name: str, key_col_def: str, extra_info: dict
    mock_get_id.side_effect = [101, 102]
    extra_info: ExtraInfo = {}
    converter._parse_auto_id(
        data=tabular_data_no_units_no_substitutions,
        table="nodes",
        ref_table=None,
        ref_name="internal_node",
        key_col_def="id_number",
        table_mask=None,
        extra_info=extra_info,
    )
    mock_get_id.assert_has_calls(
        [
            call(table="nodes", key={"id_number": 1}, name="internal_node"),
            call(table="nodes", key={"id_number": 2}, name="internal_node"),
        ]
    )
    assert extra_info[101] == {"id_reference": {"table": "nodes", "name": "internal_node", "key": {"id_number": 1}}}
    assert extra_info[102] == {"id_reference": {"table": "nodes", "name": "internal_node", "key": {"id_number": 2}}}


@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._get_id")
def test_parse_auto_id__named_keys(
    mock_get_id: MagicMock, converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData
):
    # name: str, key_col_def: Dict[str, str], extra_info: dict
    mock_get_id.side_effect = [101, 102]
    extra_info: ExtraInfo = {}
    converter._parse_auto_id(
        data=tabular_data_no_units_no_substitutions,
        table="lines",
        ref_table=None,
        ref_name=None,
        key_col_def={"id": "id_number", "node": "from_node_side"},
        table_mask=None,
        extra_info=extra_info,
    )
    mock_get_id.assert_has_calls(
        [
            call(table="lines", key={"id": 1, "node": 2}, name=None),
            call(table="lines", key={"id": 3, "node": 1}, name=None),
        ]
    )
    assert extra_info[101] == {"id_reference": {"table": "lines", "key": {"id": 1, "node": 2}}}
    assert extra_info[102] == {"id_reference": {"table": "lines", "key": {"id": 3, "node": 1}}}


def test_parse_auto_id__invalid_key_definition(
    converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData
):
    with pytest.raises(TypeError, match="Invalid key definition type 'int': 123"):
        converter._parse_auto_id(
            data=TabularData(),
            table="",
            ref_table=None,
            ref_name=None,
            key_col_def=123,  # type: ignore
            table_mask=None,
            extra_info=None,
        )


@pytest.mark.parametrize(
    ("function", "expected"),
    [
        ("multiply", (1 * 1, 2 * 10, 3 * 100)),
        ("prod", (1 * 1, 2 * 10, 3 * 100)),
        ("divide", (1 / 1, 2 / 10, 3 / 100)),
        ("sum", (1 + 1, 2 + 10, 3 + 100)),
        ("min", (1, 2, 3)),
        ("max", (1, 10, 100)),
    ],
)
@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._parse_col_def")
def test_parse_pandas_function(
    mock_parse_col_def: MagicMock,
    converter: TabularConverter,
    function: str,
    expected: Tuple[int, int, int],
):
    # Arrange
    data = MagicMock()
    col_def = ["a", "b"]
    parse_col_def_data = pd.DataFrame([[1, 1], [2, 10], [3, 100]], columns=["a", "b"])
    mock_parse_col_def.return_value = parse_col_def_data

    # Act
    result = converter._parse_pandas_function(
        data=data, table="foo", fn_name=function, col_def=col_def, table_mask=None
    )

    # Assert
    mock_parse_col_def.assert_called_once_with(
        data=data, table="foo", col_def=col_def, table_mask=None, extra_info=None
    )
    pd.testing.assert_frame_equal(result, pd.DataFrame(expected))


@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._parse_col_def")
def test_parse_pandas_function__no_data(mock_parse_col_def: MagicMock, converter: TabularConverter):
    # Arrange
    data = MagicMock()
    col_def = ["a", "b"]
    parse_col_def_data = pd.DataFrame([], columns=["a", "b"])
    mock_parse_col_def.return_value = parse_col_def_data

    # Act
    result = converter._parse_pandas_function(
        data=data, table="foo", fn_name="multiply", col_def=col_def, table_mask=None
    )

    # Assert
    mock_parse_col_def.assert_called_once_with(
        data=data, table="foo", col_def=col_def, table_mask=None, extra_info=None
    )
    assert result.empty


@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._parse_col_def")
def test_parse_pandas_function__invalid(mock_parse_col_def: MagicMock, converter: TabularConverter):
    # Arrange
    mock_parse_col_def.return_value = pd.DataFrame()

    # Act / Assert
    with pytest.raises(AssertionError):
        converter._parse_pandas_function(data=MagicMock(), table="foo", fn_name="multiply", col_def=123, table_mask=None)  # type: ignore

    # Act / Assert
    with pytest.raises(ValueError, match="Pandas DataFrame has no function 'bar'"):
        converter._parse_pandas_function(data=MagicMock(), table="foo", fn_name="bar", col_def=[], table_mask=None)

    # Act / Assert
    with pytest.raises(ValueError, match="Invalid pandas function DataFrame.apply"):
        converter._parse_pandas_function(data=MagicMock(), table="foo", fn_name="apply", col_def=[], table_mask=None)


@patch("power_grid_model_io.converters.tabular_converter.get_function")
@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._parse_col_def")
def test_parse_function(
    mock_parse_col_def: MagicMock,
    mock_get_function: MagicMock,
    converter: TabularConverter,
    tabular_data_no_units_no_substitutions: TabularData,
):
    def multiply_by_two(value: int):
        return value * 2

    mock_get_function.return_value = multiply_by_two
    mock_parse_col_def.return_value = pd.DataFrame([2, 4, 5])

    multiplied_data = converter._parse_function(
        data=tabular_data_no_units_no_substitutions,
        table="nodes",
        function="multiply_by_two",
        col_def={"value": "u_nom"},
        table_mask=None,
    )
    assert_frame_equal(multiplied_data, pd.DataFrame([4, 8, 10]))


@patch("power_grid_model_io.converters.tabular_converter.get_function")
@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._parse_col_def")
def test_parse_function__no_data(
    mock_parse_col_def: MagicMock,
    mock_get_function: MagicMock,
    converter: TabularConverter,
):
    def multiply_by_two(value: int):
        return value * 2

    mock_get_function.return_value = multiply_by_two
    mock_parse_col_def.return_value = pd.DataFrame()

    with pytest.raises(ValueError, match="multiply_by_two.*empty DataFrame"):
        converter._parse_function(
            data=TabularData(nodes=pd.DataFrame([], columns=["u_nom"])),
            table="nodes",
            function="multiply_by_two",
            col_def={"value": "u_nom"},
            table_mask=None,
        )


def test_parse_col_def_composite(converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData):
    with pytest.raises(AssertionError):
        converter._parse_col_def_composite(
            data=tabular_data_no_units_no_substitutions, table="nodes", col_def="wrong", table_mask=None  # type: ignore
        )

    df = converter._parse_col_def_composite(
        data=tabular_data_no_units_no_substitutions,
        table="nodes",
        col_def=["id_number", "u_nom"],
        table_mask=None,
    )
    assert_frame_equal(df, tabular_data_no_units_no_substitutions["nodes"])


def test_apply_multiplier__no_multipliers():
    # Arrange
    converter = TabularConverter()
    data = pd.Series([-2.0, 0.0, 2.0])

    # Act
    result = converter._apply_multiplier(table="foo", column="bar", data=data)

    # Assert
    assert converter._multipliers is None
    pd.testing.assert_series_equal(data, pd.Series([-2.0, 0.0, 2.0]))
    pd.testing.assert_series_equal(result, pd.Series([-2.0, 0.0, 2.0]))


def test_apply_multiplier():
    # Arrange
    converter = TabularConverter()
    converter._multipliers = MagicMock()
    converter._multipliers.get_multiplier.return_value = 10.0
    data = pd.Series([-2.0, 0.0, 2.0])

    # Act
    result = converter._apply_multiplier(table="foo", column="bar", data=data)

    # Assert
    converter._multipliers.get_multiplier.assert_called_once_with(table="foo", attr="bar")
    pd.testing.assert_series_equal(data, pd.Series([-2.0, 0.0, 2.0]))
    pd.testing.assert_series_equal(result, pd.Series([-20.0, 0.0, 20.0]))


def test_apply_multiplier__no_attr_multiplier():
    # Arrange
    converter = TabularConverter()
    converter._multipliers = MagicMock()
    converter._multipliers.get_multiplier.side_effect = KeyError
    data = pd.Series([-2.0, 0.0, 2.0])

    # Act
    result = converter._apply_multiplier(table="foo", column="bar", data=data)

    # Assert
    converter._multipliers.get_multiplier.assert_called_once_with(table="foo", attr="bar")
    pd.testing.assert_series_equal(data, pd.Series([-2.0, 0.0, 2.0]))
    pd.testing.assert_series_equal(result, pd.Series([-2.0, 0.0, 2.0]))


def test_get_id__private(converter: TabularConverter):
    # Arrange / Act / Assert
    assert converter._get_id(table="node", key={"a": 1, "b": 2}, name=None) == 0
    assert converter._get_id(table="node", key={"a": 1, "b": 3}, name=None) == 1  # change in values
    assert converter._get_id(table="node", key={"a": 1, "c": 2}, name=None) == 2  # change in index
    assert converter._get_id(table="foo", key={"a": 1, "b": 2}, name=None) == 3  # change in table
    assert converter._get_id(table="node", key={"a": 1, "b": 2}, name="bar") == 4  # change in name
    assert converter._get_id(table="node", key={"a": 1, "b": 2}, name=None) == 0  # duplicate name / indices / values


def test_get_id__public(converter: TabularConverter):
    # Arrange
    converter._get_id(table="node", key={"a": 1, "b": 2}, name=None)

    # Act / Assert
    assert converter.get_id(table="node", key={"a": 1, "b": 2}) == 0

    with pytest.raises(KeyError):
        converter.get_id(table="node", key={"a": 1, "b": 3})


def test_get_ids(converter: TabularConverter):
    # Arrange
    converter._get_id(table="node", key={"a": 1, "b": 2}, name=None)  # 0
    converter._get_id(table="node", key={"a": 1, "b": 3}, name=None)  # 1
    converter._get_id(table="node", key={"a": 1, "c": 2}, name=None)  # 2
    converter._get_id(table="foo", key={"a": 1, "b": 2}, name=None)  # 3
    converter._get_id(table="node", key={"a": 1, "b": 2}, name="bar")  # 4

    # Act
    query = pd.DataFrame(
        [
            {"a": 1, "b": 3},  # 1
            {"a": 1, "c": 2},  # 2
            {"a": 1, "b": 2},  # 0
        ]
    )
    pgm_ids = converter.get_ids(table="node", keys=query)

    # Assert
    assert pgm_ids == [1, 2, 0]


def test_get_ids__table_and_name(converter: TabularConverter):
    # Arrange
    converter._get_id(table="node", key={"a": 1, "b": 2}, name=None)  # 0
    converter._get_id(table="node", key={"a": 1, "b": 3}, name=None)  # 1
    converter._get_id(table="node", key={"a": 1, "c": 2}, name=None)  # 2
    converter._get_id(table="foo", key={"a": 1, "b": 2}, name=None)  # 3
    converter._get_id(table="node", key={"a": 1, "b": 2}, name="bar")  # 4

    # Act
    query = pd.DataFrame(
        [
            {"table": "node", "a": 1, "b": 3},  # 1
            {"table": "node", "a": 1, "c": 2},  # 2
            {"table": "node", "a": 1, "b": 2},  # 0
            {"table": "node", "name": "bar", "a": 1, "b": 2},  # 4
            {"table": "foo", "a": 1, "b": 2},  # 3
        ]
    )
    pgm_ids = converter.get_ids(keys=query)

    # Assert
    assert pgm_ids == [1, 2, 0, 4, 3]


def test_get_ids__round_trip(converter: TabularConverter):
    # Arrange
    converter._get_id(table="node", key={"a": 1, "b": 2}, name=None)  # 0
    converter._get_id(table="node", key={"a": 1, "b": 3}, name=None)  # 1
    converter._get_id(table="node", key={"a": 1, "c": 2}, name=None)  # 2
    converter._get_id(table="foo", key={"a": 1, "b": 2}, name=None)  # 3
    converter._get_id(table="node", key={"a": 1, "b": 2}, name="bar")  # 4

    # Act
    query = pd.DataFrame(
        [
            {"table": "node", "a": 1, "b": 3},  # 1
            {"table": "node", "a": 1, "c": 2},  # 2
            {"table": "node", "a": 1, "b": 2},  # 0
            {"table": "node", "name": "bar", "a": 1, "b": 2},  # 4
            {"table": "foo", "a": 1, "b": 2},  # 3
        ]
    )
    pgm_ids = converter.get_ids(keys=query)
    reference = converter.lookup_ids(pgm_ids=pgm_ids)

    # Assert
    # We don't know the pgm_ids in the original query, so let's remove them from the reference as well.
    # I.e. the order of the rows in query and reference should match
    reference.reset_index(drop=True, inplace=True)
    pd.testing.assert_frame_equal(reference, query)


def test_lookup_id(converter: TabularConverter):
    # Arrange
    converter._get_id(table="node", key={"a": 1, "b": 2}, name=None)
    converter._get_id(table="node", key={"a": 1, "b": 3}, name=None)  # change in values
    converter._get_id(table="node", key={"a": 1, "c": 2}, name=None)  # change in index
    converter._get_id(table="foo", key={"a": 1, "b": 2}, name=None)  # change in table
    converter._get_id(table="node", key={"a": 1, "b": 2}, name="bar")  # change in name
    converter._get_id(table="node", key={"a": 1, "b": 2}, name=None)  # duplicate name / indices / values

    # Act / Assert
    assert converter.lookup_id(pgm_id=0) == {"table": "node", "key": {"a": 1, "b": 2}}
    assert converter.lookup_id(pgm_id=1) == {"table": "node", "key": {"a": 1, "b": 3}}
    assert converter.lookup_id(pgm_id=2) == {"table": "node", "key": {"a": 1, "c": 2}}
    assert converter.lookup_id(pgm_id=3) == {"table": "foo", "key": {"a": 1, "b": 2}}
    assert converter.lookup_id(pgm_id=4) == {"table": "node", "name": "bar", "key": {"a": 1, "b": 2}}


def test_lookup_ids(converter: TabularConverter):
    # Arrange
    converter._get_id(table="node", key={"a": 1, "b": 2}, name=None)  # 0
    converter._get_id(table="node", key={"a": 1, "b": 3}, name=None)  # 1
    converter._get_id(table="node", key={"a": 1, "c": 2}, name=None)  # 2
    converter._get_id(table="foo", key={"a": 1, "b": 2}, name=None)  # 3
    converter._get_id(table="node", key={"a": 1, "b": 2}, name="bar")  # 4

    # Act
    query = [3, 2, 4]
    reference = converter.lookup_ids(pgm_ids=query)

    # Assert
    pd.testing.assert_frame_equal(
        reference,
        pd.DataFrame(
            [
                ["foo", np.nan, 1, 2, np.nan],
                ["node", np.nan, 1, np.nan, 2],
                ["node", "bar", 1, 2, np.nan],
            ],
            columns=["table", "name", "a", "b", "c"],
            index=[3, 2, 4],
        ),
        check_like=True,  # Ignore order of rows and columns
    )


def test_lookup_ids__round_trip(converter: TabularConverter):
    # Arrange
    converter._get_id(table="node", key={"a": 1, "b": 2}, name=None)  # 0
    converter._get_id(table="node", key={"a": 1, "b": 3}, name=None)  # 1
    converter._get_id(table="node", key={"a": 1, "c": 2}, name=None)  # 2
    converter._get_id(table="foo", key={"a": 1, "b": 2}, name=None)  # 3
    converter._get_id(table="node", key={"a": 1, "b": 2}, name="bar")  # 4

    # Act
    query = [3, 2, 4]
    reference = converter.lookup_ids(pgm_ids=query)
    pgm_ids = converter.get_ids(reference)

    # Assert
    assert query == pgm_ids


def test_lookup_ids__duplicate_keys(converter: TabularConverter):
    # Arrange
    converter._get_id(table="foo", name="bar", key={"table": 123, "name": 456})

    # Act
    reference = converter.lookup_ids(pgm_ids=[0])

    # Assert
    pd.testing.assert_frame_equal(reference, pd.DataFrame([[123, 456]], columns=["table", "name"], index=[0]))


@pytest.mark.parametrize(
    ("bool_fn", "expected"),
    [((True), np.array([True, True])), ((False), np.array([False, False]))],
)
@patch("power_grid_model_io.converters.tabular_converter.get_function")
def test_parse_table_filters(
    mock_get_function: MagicMock,
    converter: TabularConverter,
    tabular_data: TabularData,
    bool_fn: Callable,
    expected: np.ndarray,
):
    filtering_functions = [{"test_fn": {"kwarg_1": "a"}}]

    def bool_fn_filter(row: pd.Series, **kwargs):
        assert kwargs == {"kwarg_1": "a"}
        return bool_fn

    mock_get_function.return_value = bool_fn_filter

    actual = converter._parse_table_filters(data=tabular_data, table="nodes", filtering_functions=filtering_functions)

    mock_get_function.assert_called_once_with("test_fn")
    # check if return value is a 1d bool np array
    assert isinstance(actual, np.ndarray)
    assert actual.ndim == 1
    assert actual.dtype is np.dtype("bool")
    assert all(actual == expected)


def test_parse_table_filters__ndarray_data(converter: TabularConverter):
    numpy_tabular_data = TabularData(numpy_table=np.ones((4, 3)))
    assert converter._parse_table_filters(data=numpy_tabular_data, table="numpy_table", filtering_functions=[]) == None
