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
from power_grid_model import ComponentType, DatasetType, initialize_array, power_grid_meta_data
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
    node = initialize_array(DatasetType.input, ComponentType.node, 2)
    return {ComponentType.node: node}


@pytest.fixture
def pgm_line_empty():
    line = initialize_array(DatasetType.input, ComponentType.line, 2)
    return {ComponentType.line: line}


@pytest.fixture
def pgm_power_sensor_empty():
    power_sensor = initialize_array(DatasetType.input, ComponentType.sym_power_sensor, 1)
    return {ComponentType.sym_power_sensor: power_sensor}


@pytest.fixture
def tabular_data():
    nodes = pd.DataFrame(
        data=[[1, 10.5], [2, 0.4]],
        columns=pd.MultiIndex.from_tuples([("id_number", ""), ("u_nom", "kV")]),
    )
    lines = pd.DataFrame(data=[[1, 100], [2, 200]], columns=["id_number", "from_node_side"])
    loads = pd.DataFrame(
        data=[[1, 1, 1], [2, 2, 0]],
        columns=["id_number", "node_id", "switching_status"],
    )
    sources = pd.DataFrame(columns=["id_number", "node_side"])
    return TabularData(nodes=nodes, lines=lines, loads=loads, sources=sources)


@pytest.fixture
def tabular_data_no_units_no_substitutions() -> TabularData:
    nodes = pd.DataFrame(data=[[1, 10.5e3], [2, 400.0]], columns=["id_number", "u_nom"])
    lines = pd.DataFrame(data=[[1, 2], [3, 1]], columns=["id_number", "from_node_side"])
    sources = pd.DataFrame(columns=["id_number", "node_side"])
    return TabularData(nodes=nodes, lines=lines, sources=sources)


def test_set_mapping_file(converter: TabularConverter):
    with pytest.raises(ValueError, match="Mapping file should be a .yaml file, .txt provided."):
        converter.set_mapping_file(mapping_file=Path("dummy/path.txt"))

    dummy_path = Path(__file__).parents[2] / "data" / "config" / "dummy_mapping.yaml"
    with pytest.raises(KeyError, match="Missing 'grid' mapping in mapping_file"):
        converter.set_mapping_file(mapping_file=dummy_path)

    converter.set_mapping_file(mapping_file=MAPPING_FILE)


def test_parse_data(converter: TabularConverter, tabular_data: TabularData):
    data = MagicMock()
    converter._parse_data(data=data, data_type=MagicMock(), extra_info=None)
    data.set_unit_multipliers.assert_called_once()
    data.set_substitutions.assert_called_once()

    pgm_input_data = converter._parse_data(data=tabular_data, data_type=DatasetType.input, extra_info=None)
    assert list(pgm_input_data.keys()) == [ComponentType.node, ComponentType.line, ComponentType.sym_load]
    assert len(pgm_input_data[ComponentType.node]) == 2
    assert (pgm_input_data[ComponentType.node]["id"] == [0, 1]).all()
    assert (pgm_input_data[ComponentType.node]["u_rated"] == [10.5e3, 400]).all()

    assert len(pgm_input_data[ComponentType.line]) == 2
    assert (pgm_input_data[ComponentType.line]["id"] == [2, 3]).all()
    assert (pgm_input_data[ComponentType.line]["from_node"] == [0, 1]).all()

    assert len(pgm_input_data[ComponentType.sym_load]) == 4
    assert (pgm_input_data[ComponentType.sym_load]["id"] == [4, 5, 6, 7]).all()
    assert (pgm_input_data[ComponentType.sym_load]["node"] == [0, 1, 0, 1]).all()
    assert (pgm_input_data[ComponentType.sym_load]["status"] == [1, 0, 1, 0]).all()
    assert (
        pgm_input_data[ComponentType.sym_load].dtype
        == power_grid_meta_data[DatasetType.input][ComponentType.sym_load].dtype
    )


def test_convert_table_to_component(converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData):
    # if table does not exist in data _convert_table_to_component should return None
    none_data = converter._convert_table_to_component(
        data=tabular_data_no_units_no_substitutions,
        data_type=DatasetType.input,
        table="some_random_table",
        component=ComponentType.node,
        attributes={"key": "value"},
        extra_info=None,
    )
    assert none_data is None
    # wrong component
    with pytest.raises(KeyError, match="Invalid component type 'dummy' or data type 'input'"):
        converter._convert_table_to_component(
            data=tabular_data_no_units_no_substitutions,
            data_type=DatasetType.input,
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
            component=ComponentType.node,
            attributes={"key": "value"},
            extra_info=None,
        )
    # no 'id' in attributes
    with pytest.raises(KeyError, match="No mapping for the attribute 'id' for 'nodes'!"):
        converter._convert_table_to_component(
            data=tabular_data_no_units_no_substitutions,
            data_type=DatasetType.input,
            table="nodes",
            component=ComponentType.node,
            attributes={"key": "value"},
            extra_info=None,
        )

    node_attributes: InstanceAttributes = {"id": "id_number", "u_rated": "u_nom"}
    pgm_node_data = converter._convert_table_to_component(
        data=tabular_data_no_units_no_substitutions,
        data_type=DatasetType.input,
        table="nodes",
        component=ComponentType.node,
        attributes=node_attributes,
        extra_info=None,
    )
    assert pgm_node_data is not None
    assert len(pgm_node_data) == 2
    assert (pgm_node_data["id"] == [1, 2]).all()
    assert (pgm_node_data["u_rated"] == [10.5e3, 400]).all()


@patch(
    "power_grid_model_io.converters.tabular_converter.TabularConverter._convert_col_def_to_attribute",
    new=MagicMock(),
)
@patch(
    "power_grid_model_io.converters.tabular_converter.TabularConverter._parse_table_filters",
)
def test_convert_table_to_component__filters(
    _parse_table_filters: MagicMock,
    converter: TabularConverter,
    tabular_data_no_units_no_substitutions: TabularData,
):
    node_attributes_with_filter: dict[str, int | float | str | dict | list] = {
        "id": "id_number",
        "u_rated": "u_nom",
        "filters": [{"test_fn": {}}],
    }
    converter._convert_table_to_component(
        data=tabular_data_no_units_no_substitutions,
        data_type=DatasetType.input,
        table="nodes",
        component=ComponentType.node,
        attributes=node_attributes_with_filter,
        extra_info=None,
    )
    _parse_table_filters.assert_called_once_with(
        data=tabular_data_no_units_no_substitutions,
        table="nodes",
        filtering_functions=node_attributes_with_filter["filters"],
    )


@patch(
    "power_grid_model_io.converters.tabular_converter.TabularConverter._convert_col_def_to_attribute",
)
@patch(
    "power_grid_model_io.converters.tabular_converter.TabularConverter._parse_table_filters",
    side_effect=np.array([False, False]),
)
def test_convert_table_to_component__filters_all_false(
    _parse_table_filters: MagicMock,
    _convert_col_def_to_attribute: MagicMock,
    converter: TabularConverter,
    tabular_data_no_units_no_substitutions: TabularData,
):
    node_attributes_with_filter: dict[str, int | float | str | dict | list] = {
        "id": "id_number",
        "u_rated": "u_nom",
        "filters": [{"test_fn": {}}],
    }
    actual = converter._convert_table_to_component(
        data=tabular_data_no_units_no_substitutions,
        data_type=DatasetType.input,
        table="nodes",
        component=ComponentType.node,
        attributes=node_attributes_with_filter,
        extra_info=None,
    )

    assert actual is None
    _parse_table_filters.assert_called_once_with(
        data=tabular_data_no_units_no_substitutions,
        table="nodes",
        filtering_functions=node_attributes_with_filter["filters"],
    )
    _convert_col_def_to_attribute.assert_not_called()


def test_convert_col_def_to_attribute(
    converter: TabularConverter,
    tabular_data_no_units_no_substitutions: TabularData,
    pgm_node_empty: SingleDataset,
):
    with pytest.raises(
        KeyError,
        match=r"Could not find attribute 'incorrect_attribute' for 'nodes'. \(choose from: id, u_rated\)",
    ):
        converter._convert_col_def_to_attribute(
            data=tabular_data_no_units_no_substitutions,
            pgm_data=pgm_node_empty[ComponentType.node],
            table="nodes",
            component=ComponentType.node,
            attr="incorrect_attribute",
            col_def="id_number",
            table_mask=None,
            extra_info=None,
        )

    # test extra info
    converter._convert_col_def_to_attribute(
        data=tabular_data_no_units_no_substitutions,
        pgm_data=pgm_node_empty[ComponentType.node],
        table="nodes",
        component=ComponentType.node,
        attr="extra",
        col_def="u_nom",
        table_mask=None,
        extra_info={0: {}, 1: {}},
    )

    # test other attr
    converter._convert_col_def_to_attribute(
        data=tabular_data_no_units_no_substitutions,
        pgm_data=pgm_node_empty[ComponentType.node],
        table="nodes",
        component=ComponentType.node,
        attr="u_rated",
        col_def="u_nom",
        table_mask=None,
        extra_info=None,
    )
    assert len(pgm_node_empty) == 1
    assert (pgm_node_empty[ComponentType.node]["u_rated"] == [10500.0, 400.0]).all()

    with pytest.raises(
        ValueError,
        match=r"DataFrame for ComponentType.node.u_rated should contain a single column "
        r"\(Index\(\['id_number', 'u_nom'\], dtype='object'\)\)",
    ):
        converter._convert_col_def_to_attribute(
            data=tabular_data_no_units_no_substitutions,
            pgm_data=pgm_node_empty[ComponentType.node],
            table="nodes",
            component=ComponentType.node,
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
    nodes_1 = initialize_array(DatasetType.input, ComponentType.node, 2)
    nodes_1["id"] = [0, 1]

    nodes_2 = initialize_array(DatasetType.input, ComponentType.node, 3)
    nodes_2["id"] = [2, 3, 4]
    data = {ComponentType.node: [nodes_1, nodes_2]}

    merged = converter._merge_pgm_data(data)
    assert len(merged) == 1
    assert (merged[ComponentType.node]["id"] == np.array([0, 1, 2, 3, 4])).all()


def test_serialize_data(converter: TabularConverter, pgm_node_empty: SingleDataset):
    with pytest.raises(
        NotImplementedError,
        match=r"Extra info can not \(yet\) be stored for tabular data",
    ):
        converter._serialize_data(data=pgm_node_empty, extra_info={})
    with pytest.raises(
        NotImplementedError,
        match=r"Batch data can not \(yet\) be stored for tabular data",
    ):
        converter._serialize_data(data=[], extra_info=None)  # type: ignore

    pgm_node_empty[ComponentType.node]["id"] = [1, 2]
    pgm_node_empty[ComponentType.node]["u_rated"] = [3.0, 4.0]
    tabular_data = converter._serialize_data(data=pgm_node_empty, extra_info=None)
    assert len(tabular_data) == 1
    assert (tabular_data[ComponentType.node]["id"] == np.array([1, 2])).all()
    assert (tabular_data[ComponentType.node]["u_rated"] == np.array([3.0, 4.0])).all()


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
            data=tabular_data_no_units_no_substitutions,
            table="nodes",
            table_mask=None,
            col_def=50,
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
            data=tabular_data_no_units_no_substitutions,
            table="nodes",
            col_def=4.0,
            table_mask=None,
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
            data=tabular_data_no_units_no_substitutions,
            table="nodes",
            col_def="col_name",
            table_mask=None,
            allow_missing=False,
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
            data=tabular_data_no_units_no_substitutions,
            table="nodes",
            col_def=[],
            table_mask=None,
            allow_missing=False,
        )


def test_parse_col_def_const(converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData):
    with pytest.raises(AssertionError):
        converter._parse_col_def_const(
            data=tabular_data_no_units_no_substitutions,
            table="nodes",
            col_def="str",  # type: ignore
            table_mask=None,  # type: ignore
        )

    # type(col_def) == int
    col_int = converter._parse_col_def_const(
        data=tabular_data_no_units_no_substitutions,
        table="nodes",
        col_def=50,
        table_mask=None,
    )
    assert_frame_equal(col_int, pd.DataFrame([50, 50]))

    # type(col_def) == float
    col_int = converter._parse_col_def_const(
        data=tabular_data_no_units_no_substitutions,
        table="nodes",
        col_def=3.0,
        table_mask=None,
    )
    assert_frame_equal(col_int, pd.DataFrame([3.0, 3.0]))


def test_parse_col_def_const__no_filter(
    converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData
):
    col_int = converter._parse_col_def_const(
        data=tabular_data_no_units_no_substitutions,
        table="nodes",
        col_def=3.0,
        table_mask=None,
    )
    assert_frame_equal(col_int, pd.DataFrame([3.0, 3.0]))


def test_parse_col_def_column_name(converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData):
    with pytest.raises(AssertionError):
        converter._parse_col_def_column_name(
            data=tabular_data_no_units_no_substitutions,
            table="nodes",
            col_def=1,  # type: ignore
            table_mask=None,  # type: ignore
        )

    df_multiple_columns = converter._parse_col_def_column_name(
        data=tabular_data_no_units_no_substitutions,
        table="nodes",
        col_def="  wrong_column  | id_number  | u_nom  ",
        table_mask=None,
    )
    assert_frame_equal(df_multiple_columns, pd.DataFrame([1, 2], columns=["id_number"]))

    df_inf = converter._parse_col_def_column_name(
        data=tabular_data_no_units_no_substitutions,
        table="nodes",
        col_def="inf",
        table_mask=None,
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


def test_parse_col_def_filter(converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData):
    # Act/Assert:
    with pytest.raises(AssertionError):
        converter._parse_col_def_filter(
            data=tabular_data_no_units_no_substitutions,
            table="",
            col_def=[],  # type: ignore
            table_mask=None,
            extra_info=None,  # type: ignore
        )

    with pytest.raises(TypeError, match="Invalid foo definition: 123"):
        converter._parse_col_def_filter(
            data=MagicMock(),
            table="",
            col_def={"foo": 123},
            table_mask=None,
            extra_info=None,
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
        data=data,
        table="nodes",
        col_def={"multiply": ["id_number", "u_nom"]},
        table_mask=None,
        extra_info=None,
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
            data=data,
            table="",
            col_def={"auto_id": {"a": 1, "b": 2}},
            table_mask=None,
            extra_info=None,
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
            data=data,
            table="",
            col_def={"reference": {"a": 1, "b": 2}},
            table_mask=None,
            extra_info=None,
        )


@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._get_id")
def test_parse_auto_id(
    mock_get_id: MagicMock,
    converter: TabularConverter,
    tabular_data_no_units_no_substitutions: TabularData,
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
        [
            call(table="nodes", key={"id_number": 1}, name=None),
            call(table="nodes", key={"id_number": 2}, name=None),
        ]
    )


@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._get_id")
def test_parse_auto_id__extra_info(
    mock_get_id: MagicMock,
    converter: TabularConverter,
    tabular_data_no_units_no_substitutions: TabularData,
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
        [
            call(table="nodes", key={"id_number": 1}, name=None),
            call(table="nodes", key={"id_number": 2}, name=None),
        ]
    )
    assert extra_info[101] == {"id_reference": {"table": "nodes", "key": {"id_number": 1}}}
    assert extra_info[102] == {"id_reference": {"table": "nodes", "key": {"id_number": 2}}}


@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._get_id")
def test_parse_auto_id__reference_column(
    mock_get_id: MagicMock,
    converter: TabularConverter,
    tabular_data_no_units_no_substitutions: TabularData,
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
        [
            call(table="nodes", key={"id_number": 2}, name=None),
            call(table="nodes", key={"id_number": 1}, name=None),
        ]
    )
    assert len(extra_info) == 0


@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._get_id")
def test_parse_auto_id__composite_key(
    mock_get_id: MagicMock,
    converter: TabularConverter,
    tabular_data_no_units_no_substitutions: TabularData,
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
    mock_get_id: MagicMock,
    converter: TabularConverter,
    tabular_data_no_units_no_substitutions: TabularData,
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
    assert extra_info[101] == {
        "id_reference": {
            "table": "nodes",
            "name": "internal_node",
            "key": {"id_number": 1},
        }
    }
    assert extra_info[102] == {
        "id_reference": {
            "table": "nodes",
            "name": "internal_node",
            "key": {"id_number": 2},
        }
    }


@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._get_id")
def test_parse_auto_id__named_keys(
    mock_get_id: MagicMock,
    converter: TabularConverter,
    tabular_data_no_units_no_substitutions: TabularData,
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


@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._get_id")
def test_parse_auto_id__empty_col_data(
    mock_get_id: MagicMock,
    converter: TabularConverter,
    tabular_data_no_units_no_substitutions: TabularData,
):
    converter._parse_auto_id(
        data=tabular_data_no_units_no_substitutions,
        table="lines",
        ref_table=None,
        ref_name=None,
        key_col_def={"id": "id_number", "node": "from_node_side"},
        table_mask=np.array([False, False]),
        extra_info={},
    )
    mock_get_id.assert_not_called()


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
        converter._parse_pandas_function(
            data=MagicMock(),
            table="foo",
            fn_name="multiply",
            col_def=123,  # type: ignore
            table_mask=None,
        )  # type: ignore

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
            data=tabular_data_no_units_no_substitutions,
            table="nodes",
            col_def="wrong",  # type: ignore
            table_mask=None,  # type: ignore
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
    assert converter._parse_table_filters(data=numpy_tabular_data, table="numpy_table", filtering_functions=[]) is None


def test_optional_extra__all_columns_present(converter: TabularConverter):
    """Test optional_extra when all optional columns are present in the data"""
    # Arrange
    data = TabularData(
        test_table=pd.DataFrame(
            {"id": [1, 2], "name": ["node1", "node2"], "guid": ["guid1", "guid2"], "station": ["st1", "st2"]}
        )
    )
    col_def = {"optional_extra": ["guid", "station"]}

    # Act
    result = converter._parse_col_def(
        data=data, table="test_table", col_def=col_def, table_mask=None, extra_info=None, allow_missing=False
    )

    # Assert
    assert list(result.columns) == ["guid", "station"]
    assert list(result["guid"]) == ["guid1", "guid2"]
    assert list(result["station"]) == ["st1", "st2"]


def test_optional_extra__some_columns_missing(converter: TabularConverter):
    """Test optional_extra when some optional columns are missing from the data"""
    # Arrange
    data = TabularData(test_table=pd.DataFrame({"id": [1, 2], "name": ["node1", "node2"], "guid": ["guid1", "guid2"]}))
    col_def = {"optional_extra": ["guid", "station"]}  # 'station' is missing

    # Act
    result = converter._parse_col_def(
        data=data, table="test_table", col_def=col_def, table_mask=None, extra_info=None, allow_missing=False
    )

    # Assert - only 'guid' should be present
    assert list(result.columns) == ["guid"]
    assert list(result["guid"]) == ["guid1", "guid2"]


def test_optional_extra__all_columns_missing(converter: TabularConverter):
    """Test optional_extra when all optional columns are missing from the data"""
    # Arrange
    data = TabularData(test_table=pd.DataFrame({"id": [1, 2], "name": ["node1", "node2"]}))
    col_def = {"optional_extra": ["guid", "station"]}  # Both are missing

    # Act
    result = converter._parse_col_def(
        data=data, table="test_table", col_def=col_def, table_mask=None, extra_info=None, allow_missing=False
    )

    # Assert - should return empty DataFrame with correct number of rows
    assert len(result) == 2
    assert len(result.columns) == 0


def test_optional_extra__mixed_with_required(converter: TabularConverter):
    """Test mixing required and optional extra columns"""
    # Arrange
    data = TabularData(test_table=pd.DataFrame({"id": [1, 2], "name": ["node1", "node2"], "guid": ["guid1", "guid2"]}))
    # Mix required columns with optional_extra
    col_def = ["name", {"optional_extra": ["guid", "station"]}]

    # Act
    result = converter._parse_col_def(
        data=data, table="test_table", col_def=col_def, table_mask=None, extra_info=None, allow_missing=False
    )

    # Assert - should have 'name' and 'guid', but not 'station'
    assert list(result.columns) == ["name", "guid"]
    assert list(result["name"]) == ["node1", "node2"]
    assert list(result["guid"]) == ["guid1", "guid2"]


def test_optional_extra__mixed_with_required_missing_column(converter: TabularConverter):
    """Test that missing required column raises error even with optional_extra present"""
    # Arrange
    data = TabularData(test_table=pd.DataFrame({"id": [1, 2], "name": ["node1", "node2"], "guid": ["guid1", "guid2"]}))
    # "missing_col" is required but missing
    col_def = ["name", "missing_col", {"optional_extra": ["guid", "station"]}]

    # Act & Assert
    with pytest.raises(KeyError, match="missing_col"):
        converter._parse_col_def(
            data=data, table="test_table", col_def=col_def, table_mask=None, extra_info=None, allow_missing=False
        )


def test_optional_extra__in_extra_info(converter: TabularConverter):
    """Test that optional_extra works correctly with _handle_extra_info"""
    # Arrange
    data = TabularData(test_table=pd.DataFrame({"id": [1, 2], "name": ["node1", "node2"], "guid": ["guid1", "guid2"]}))
    uuids = np.array([100, 200])
    extra_info: ExtraInfo = {}
    col_def = {"optional_extra": ["guid", "station"]}

    # Act
    converter._handle_extra_info(
        data=data, table="test_table", col_def=col_def, uuids=uuids, table_mask=None, extra_info=extra_info
    )

    # Assert - only 'guid' should be in extra_info, not 'station'
    assert 100 in extra_info
    assert 200 in extra_info
    assert "guid" in extra_info[100]
    assert "guid" in extra_info[200]
    assert extra_info[100]["guid"] == "guid1"
    assert extra_info[200]["guid"] == "guid2"
    assert "station" not in extra_info[100]
    assert "station" not in extra_info[200]


def test_optional_extra__empty_list_with_extra_info(converter: TabularConverter):
    """Test that empty optional_extra list results in no updates to extra_info"""
    # Arrange
    data = TabularData(test_table=pd.DataFrame({"id": [1, 2], "name": ["node1", "node2"], "guid": ["guid1", "guid2"]}))
    uuids = np.array([100, 200])
    extra_info: ExtraInfo = {}
    col_def: dict[str, list[str]] = {"optional_extra": []}

    # Act
    converter._handle_extra_info(
        data=data, table="test_table", col_def=col_def, uuids=uuids, table_mask=None, extra_info=extra_info
    )

    # Assert
    assert len(extra_info) == 0


def test_optional_extra__all_missing_no_extra_info(converter: TabularConverter):
    """Test that when all optional columns are missing, no extra_info entries are created"""
    # Arrange
    data = TabularData(test_table=pd.DataFrame({"id": [1, 2], "name": ["node1", "node2"]}))  # Both optional missing
    uuids = np.array([100, 200])
    extra_info: ExtraInfo = {}
    col_def = {"optional_extra": ["guid", "station"]}

    # Act
    converter._handle_extra_info(
        data=data, table="test_table", col_def=col_def, uuids=uuids, table_mask=None, extra_info=extra_info
    )

    # Assert - no entries should be added to extra_info
    assert len(extra_info) == 0


def test_optional_extra__invalid_type():
    """Test that optional_extra raises TypeError if value is not a list"""
    # Arrange
    converter = TabularConverter(mapping_file=MAPPING_FILE)
    data = TabularData(test_table=pd.DataFrame({"id": [1, 2]}))
    col_def = {"optional_extra": "not_a_list"}  # Invalid: should be a list

    # Act & Assert
    with pytest.raises(TypeError, match="optional_extra value must be a list"):
        converter._parse_col_def(
            data=data, table="test_table", col_def=col_def, table_mask=None, extra_info=None, allow_missing=False
        )


def test_optional_extra__integration():
    """Integration test for optional_extra feature using a complete mapping file"""
    # Arrange
    mapping_file = Path(__file__).parents[2] / "data" / "config" / "test_optional_extra_mapping.yaml"
    converter = TabularConverter(mapping_file=mapping_file)

    # Create test data with some optional columns present and some missing
    data = TabularData(
        nodes=pd.DataFrame(
            {
                "node_id": [1, 2, 3],
                "voltage": [10.5, 10.5, 0.4],
                "ID": ["N1", "N2", "N3"],
                "Name": ["Node 1", "Node 2", "Node 3"],
                "GUID": ["guid-1", "guid-2", "guid-3"],
                # Note: StationID column is missing (optional)
            }
        )
    )

    extra_info: ExtraInfo = {}

    # Act
    result = converter._parse_data(data=data, data_type=DatasetType.input, extra_info=extra_info)

    # Assert
    assert ComponentType.node in result
    assert len(result[ComponentType.node]) == 3

    # Check that extra_info contains the required and present optional fields
    for node_id in result[ComponentType.node]["id"]:
        assert node_id in extra_info
        assert "ID" in extra_info[node_id]
        assert "Name" in extra_info[node_id]
        assert "GUID" in extra_info[node_id]  # Optional but present
        assert "StationID" not in extra_info[node_id]  # Optional and missing

    # Verify values
    node_0_id = result[ComponentType.node]["id"][0]
    assert extra_info[node_0_id]["ID"] == "N1"
    assert extra_info[node_0_id]["Name"] == "Node 1"
    assert extra_info[node_0_id]["GUID"] == "guid-1"


def test_optional_extra__with_table_mask(converter: TabularConverter):
    """Test optional_extra works correctly with table filtering/masking"""
    # Arrange
    data = TabularData(
        test_table=pd.DataFrame(
            {
                "id": [1, 2, 3, 4],
                "value": [10, 20, 30, 40],
                "guid": ["g1", "g2", "g3", "g4"],
                "name": ["n1", "n2", "n3", "n4"],
            }
        )
    )
    # Create a mask that filters to only rows 0 and 2
    table_mask = np.array([True, False, True, False])
    col_def = {"optional_extra": ["guid", "station"]}  # 'station' is missing

    # Act
    result = converter._parse_col_def(
        data=data, table="test_table", col_def=col_def, table_mask=table_mask, extra_info=None, allow_missing=False
    )

    # Assert - should only have 2 rows (from the mask) and 1 column (guid)
    assert len(result) == 2
    assert list(result.columns) == ["guid"]
    assert list(result["guid"]) == ["g1", "g3"]


def test_optional_extra__nested_in_list(converter: TabularConverter):
    """Test optional_extra can be nested within a regular list of columns"""
    # Arrange
    data = TabularData(
        test_table=pd.DataFrame(
            {"id": [1, 2], "name": ["n1", "n2"], "value": [100, 200], "guid": ["g1", "g2"]}  # station missing
        )
    )
    col_def = ["name", "value", {"optional_extra": ["guid", "station"]}]

    # Act
    result = converter._parse_col_def(
        data=data, table="test_table", col_def=col_def, table_mask=None, extra_info=None, allow_missing=False
    )

    # Assert
    assert list(result.columns) == ["name", "value", "guid"]
    assert list(result["name"]) == ["n1", "n2"]
    assert list(result["value"]) == [100, 200]
    assert list(result["guid"]) == ["g1", "g2"]


def test_optional_extra__with_pipe_separated_columns(converter: TabularConverter):
    """Test optional_extra with pipe-separated alternative column names"""
    # Arrange
    data = TabularData(test_table=pd.DataFrame({"id": [1, 2], "GUID": ["g1", "g2"], "name": ["n1", "n2"]}))
    # Use pipe separator for alternative column names (GUID or Guid)
    col_def = {"optional_extra": ["GUID|Guid", "StationID|Station"]}  # Both StationID and Station missing

    # Act
    result = converter._parse_col_def(
        data=data, table="test_table", col_def=col_def, table_mask=None, extra_info=None, allow_missing=False
    )

    # Assert - GUID should be found, Station alternatives should be skipped
    assert list(result.columns) == ["GUID"]
    assert list(result["GUID"]) == ["g1", "g2"]


def test_optional_extra__empty_string_values(converter: TabularConverter):
    """Test that optional_extra handles empty strings correctly"""
    # Arrange
    data = TabularData(test_table=pd.DataFrame({"id": [1, 2, 3], "guid": ["g1", "", "g3"], "name": ["n1", "n2", ""]}))
    uuids = np.array([100, 200, 300])
    extra_info: ExtraInfo = {}
    col_def = {"optional_extra": ["guid", "name"]}

    # Act
    converter._handle_extra_info(
        data=data, table="test_table", col_def=col_def, uuids=uuids, table_mask=None, extra_info=extra_info
    )

    # Assert - empty strings should still be included (not filtered as NaN)
    assert 100 in extra_info
    assert 200 in extra_info
    assert 300 in extra_info
    assert extra_info[100]["guid"] == "g1"
    assert extra_info[100]["name"] == "n1"
    assert extra_info[200]["guid"] == ""  # Empty string preserved
    assert extra_info[200]["name"] == "n2"
    assert extra_info[300]["guid"] == "g3"
    assert extra_info[300]["name"] == ""  # Empty string preserved


def test_optional_extra__with_nan_values(converter: TabularConverter):
    """Test that optional_extra filters out NaN values correctly"""
    # Arrange
    data = TabularData(
        test_table=pd.DataFrame({"id": [1, 2, 3], "guid": ["g1", np.nan, "g3"], "value": [10.0, 20.0, np.nan]})
    )
    uuids = np.array([100, 200, 300])
    extra_info: ExtraInfo = {}
    col_def = {"optional_extra": ["guid", "value"]}

    # Act
    converter._handle_extra_info(
        data=data, table="test_table", col_def=col_def, uuids=uuids, table_mask=None, extra_info=extra_info
    )

    # Assert - NaN values should be filtered out
    assert 100 in extra_info
    assert extra_info[100] == {"guid": "g1", "value": 10.0}

    assert 200 in extra_info
    assert extra_info[200] == {"value": 20.0}  # guid was NaN, filtered out

    assert 300 in extra_info
    assert extra_info[300] == {"guid": "g3"}  # value was NaN, filtered out


def test_optional_extra__multiple_optional_extra_sections():
    """Test behavior when multiple optional_extra sections are used (should work independently)"""
    # Arrange
    converter = TabularConverter(mapping_file=MAPPING_FILE)
    data = TabularData(
        test_table=pd.DataFrame(
            {"id": [1, 2], "name": ["node1", "node2"], "guid": ["guid1", "guid2"]}  # station and zone missing
        )
    )
    # Two separate optional_extra sections
    col_def = [{"optional_extra": ["guid"]}, {"optional_extra": ["station", "zone"]}]

    # Act
    result = converter._parse_col_def(
        data=data, table="test_table", col_def=col_def, table_mask=None, extra_info=None, allow_missing=False
    )

    # Assert - only guid should be present
    assert list(result.columns) == ["guid"]
    assert list(result["guid"]) == ["guid1", "guid2"]


def test_convert_col_def_to_attribute__pgm_data_without_dtype_names():
    """Test error handling when pgm_data has no dtype.names (unusual edge case)"""
    # Arrange
    converter = TabularConverter(mapping_file=MAPPING_FILE)
    data = TabularData(test_table=pd.DataFrame({"id": [1, 2], "name": ["n1", "n2"]}))

    # Create a mock array without dtype.names by using a plain ndarray
    pgm_data = np.array([1, 2])  # Regular array without structured dtype
    assert pgm_data.dtype.names is None

    # Act & Assert
    with pytest.raises(ValueError, match="pgm_data for 'nodes' has no attributes defined"):
        converter._convert_col_def_to_attribute(
            data=data,
            pgm_data=pgm_data,
            table="test_table",
            component="node",
            attr="id",
            col_def="id",
            table_mask=None,
            extra_info=None,
        )


def test_parse_col_def_with_allow_missing():
    """Test _parse_col_def function with allow_missing parameter both True and False"""
    # Arrange
    converter = TabularConverter(mapping_file=MAPPING_FILE)
    data = TabularData(test_table=pd.DataFrame({"existing_col": [1, 2, 3], "another_col": ["a", "b", "c"]}))

    # Test 1: String column with allow_missing=False (default) - existing column
    result = converter._parse_col_def(
        data=data,
        table="test_table",
        col_def="existing_col",
        table_mask=None,
        extra_info=None,
        allow_missing=False,
    )
    assert list(result.iloc[:, 0]) == [1, 2, 3]

    # Test 2: String column with allow_missing=False - missing column (should raise KeyError)
    with pytest.raises(KeyError, match="Could not find column 'missing_col' on table 'test_table'"):
        converter._parse_col_def(
            data=data,
            table="test_table",
            col_def="missing_col",
            table_mask=None,
            extra_info=None,
            allow_missing=False,
        )

    # Test 3: String column with allow_missing=True - missing column (should return empty DataFrame)
    result = converter._parse_col_def(
        data=data,
        table="test_table",
        col_def="missing_col",
        table_mask=None,
        extra_info=None,
        allow_missing=True,
    )
    assert len(result.columns) == 0
    assert len(result) == 3  # Should have same number of rows as original table

    # Test 4: String column with allow_missing=True - existing column (should work normally)
    result = converter._parse_col_def(
        data=data,
        table="test_table",
        col_def="existing_col",
        table_mask=None,
        extra_info=None,
        allow_missing=True,
    )
    assert list(result.iloc[:, 0]) == [1, 2, 3]

    # Test 5: List (composite) with allow_missing=False - all existing columns
    result = converter._parse_col_def(
        data=data,
        table="test_table",
        col_def=["existing_col", "another_col"],
        table_mask=None,
        extra_info=None,
        allow_missing=False,
    )
    assert len(result.columns) == 2
    assert list(result["existing_col"]) == [1, 2, 3]
    assert list(result["another_col"]) == ["a", "b", "c"]

    # Test 6: List (composite) with allow_missing=False - some missing columns (should raise error)
    with pytest.raises(KeyError, match="Could not find column 'missing_col' on table 'test_table'"):
        converter._parse_col_def(
            data=data,
            table="test_table",
            col_def=["existing_col", "missing_col"],
            table_mask=None,
            extra_info=None,
            allow_missing=False,
        )

    # Test 7: List (composite) with allow_missing=True - some missing columns (should skip missing)
    result = converter._parse_col_def(
        data=data,
        table="test_table",
        col_def=["existing_col", "missing_col", "another_col"],
        table_mask=None,
        extra_info=None,
        allow_missing=True,
    )
    assert len(result.columns) == 2  # Only existing columns should be present
    assert list(result["existing_col"]) == [1, 2, 3]
    assert list(result["another_col"]) == ["a", "b", "c"]

    # Test 8: List (composite) with allow_missing=True - all missing columns (should return empty with correct rows)
    result = converter._parse_col_def(
        data=data,
        table="test_table",
        col_def=["missing_col1", "missing_col2"],
        table_mask=None,
        extra_info=None,
        allow_missing=True,
    )
    assert len(result.columns) == 0
    assert len(result) == 3  # Should have same number of rows as original table

    # Test 9: Dict (optional_extra) - should automatically set allow_missing=True internally
    result = converter._parse_col_def(
        data=data,
        table="test_table",
        col_def={"optional_extra": ["existing_col", "missing_col"]},
        table_mask=None,
        extra_info=None,
        allow_missing=False,  # This should be ignored for optional_extra
    )
    assert len(result.columns) == 1  # Only existing column should be present
    assert list(result["existing_col"]) == [1, 2, 3]

    # Test 10: Constant values should work regardless of allow_missing
    result_false = converter._parse_col_def(
        data=data,
        table="test_table",
        col_def=42,
        table_mask=None,
        extra_info=None,
        allow_missing=False,
    )
    result_true = converter._parse_col_def(
        data=data,
        table="test_table",
        col_def=42,
        table_mask=None,
        extra_info=None,
        allow_missing=True,
    )
    assert list(result_false.iloc[:, 0]) == [42, 42, 42]
    assert list(result_true.iloc[:, 0]) == [42, 42, 42]


def test_parse_col_def_with_allow_missing_and_table_mask():
    """Test _parse_col_def function with allow_missing and table_mask combinations"""
    # Arrange
    converter = TabularConverter(mapping_file=MAPPING_FILE)
    data = TabularData(test_table=pd.DataFrame({"existing_col": [1, 2, 3, 4], "another_col": ["a", "b", "c", "d"]}))
    table_mask = np.array([True, False, True, False])  # Select rows 0 and 2

    # Test 1: Missing column with table_mask and allow_missing=True
    result = converter._parse_col_def(
        data=data,
        table="test_table",
        col_def="missing_col",
        table_mask=table_mask,
        extra_info=None,
        allow_missing=True,
    )
    assert len(result.columns) == 0
    assert len(result) == 2  # Should match filtered table length

    # Test 2: Existing column with table_mask and allow_missing=True
    result = converter._parse_col_def(
        data=data,
        table="test_table",
        col_def="existing_col",
        table_mask=table_mask,
        extra_info=None,
        allow_missing=True,
    )
    assert list(result.iloc[:, 0]) == [1, 3]  # Should get filtered values

    # Test 3: Composite with missing columns, table_mask, and allow_missing=True
    result = converter._parse_col_def(
        data=data,
        table="test_table",
        col_def=["existing_col", "missing_col"],
        table_mask=table_mask,
        extra_info=None,
        allow_missing=True,
    )
    assert len(result.columns) == 1  # Only existing column
    assert list(result["existing_col"]) == [1, 3]  # Filtered values


def test_normalize_extra_col_def():
    """Test _normalize_extra_col_def method for handling duplicate columns"""
    # Arrange
    converter = TabularConverter(mapping_file=MAPPING_FILE)

    # Test 1: Regular list without optional_extra (should be unchanged)
    col_def = ["ID", "Name", "GUID"]
    result = converter._normalize_extra_col_def(col_def)
    assert result == ["ID", "Name", "GUID"]

    # Test 2: Non-list input (should be unchanged)
    col_def = "ID"
    result = converter._normalize_extra_col_def(col_def)
    assert result == "ID"

    # Test 3: List with optional_extra but no duplicates
    col_def = ["ID", {"optional_extra": ["GUID", "StationID"]}]
    result = converter._normalize_extra_col_def(col_def)
    assert result == ["ID", {"optional_extra": ["GUID", "StationID"]}]

    # Test 4: List with duplicates - regular column should dominate
    col_def = ["ID", "Name", {"optional_extra": ["ID", "GUID", "StationID"]}]
    result = converter._normalize_extra_col_def(col_def)
    expected = ["ID", "Name", {"optional_extra": ["GUID", "StationID"]}]
    assert result == expected

    # Test 5: Multiple optional_extra sections with overlaps
    col_def = ["ID", {"optional_extra": ["ID", "GUID"]}, {"optional_extra": ["Name", "StationID"]}]
    result = converter._normalize_extra_col_def(col_def)
    expected = ["ID", {"optional_extra": ["GUID"]}, {"optional_extra": ["Name", "StationID"]}]
    assert result == expected

    # Test 6: All optional columns are duplicates (should remove optional_extra section)
    col_def = ["ID", "Name", {"optional_extra": ["ID", "Name"]}]
    result = converter._normalize_extra_col_def(col_def)
    expected = ["ID", "Name"]
    assert result == expected

    # Test 7: Empty optional_extra list (should be removed)
    col_def = ["ID", {"optional_extra": []}]
    result = converter._normalize_extra_col_def(col_def)
    expected = ["ID"]
    assert result == expected


def test_handle_extra_info_with_duplicates():
    """Test that _handle_extra_info correctly handles duplicates between regular and optional columns"""
    # Arrange
    converter = TabularConverter(mapping_file=MAPPING_FILE)
    data = TabularData(
        test_table=pd.DataFrame(
            {
                "ID": ["N001", "N002", "N003"],
                "Name": ["Node1", "Node2", "Node3"],
                "GUID": ["g1", "g2", "g3"],
                "StationID": ["ST1", "ST2", "ST3"],
            }
        )
    )

    # Column definition with duplicates (ID appears in both regular and optional_extra)
    col_def = ["ID", "Name", {"optional_extra": ["ID", "GUID", "StationID"]}]

    uuids = np.array([100, 200, 300])
    extra_info = {}

    # Act
    converter._handle_extra_info(
        data=data,
        table="test_table",
        col_def=col_def,
        uuids=uuids,
        table_mask=None,
        extra_info=extra_info,
    )

    # Assert
    # ID should appear only once (from regular column, not duplicated from optional_extra)
    assert 100 in extra_info
    assert extra_info[100]["ID"] == "N001"
    assert extra_info[100]["Name"] == "Node1"
    assert extra_info[100]["GUID"] == "g1"
    assert extra_info[100]["StationID"] == "ST1"

    # Check that we don't have duplicate ID columns in the result
    result_keys = list(extra_info[100].keys())
    assert result_keys.count("ID") == 1, f"ID should appear only once, but got: {result_keys}"

    # Similar checks for other rows
    assert extra_info[200]["ID"] == "N002"
    assert extra_info[300]["ID"] == "N003"


def test_optional_extra_with_duplicates_integration():
    """Integration test to verify duplicate elimination works in a full conversion scenario"""
    # Arrange
    converter = TabularConverter(mapping_file=MAPPING_FILE)

    # Create test data with columns that will appear in both regular and optional_extra
    data = TabularData(
        test_table=pd.DataFrame(
            {
                "id": [1, 2, 3],
                "name": ["Node1", "Node2", "Node3"],
                "u_nom": [10.0, 10.0, 0.4],
                "guid": ["g1", "g2", "g3"],
                "station": ["ST1", "ST2", "ST3"],
            }
        )
    )

    # Column definition that has ID in both places - regular should dominate
    col_def = [
        "id",  # Regular column
        "name",  # Regular column
        {"optional_extra": ["id", "guid", "station"]},  # ID is duplicate, guid and station are new
    ]

    extra_info = {}
    uuids = np.array([100, 200, 300])

    # Act
    converter._handle_extra_info(
        data=data,
        table="test_table",
        col_def=col_def,
        uuids=uuids,
        table_mask=None,
        extra_info=extra_info,
    )

    # Assert - verify no duplicate columns and all expected columns are present
    for uuid, expected_id in zip([100, 200, 300], [1, 2, 3]):
        assert uuid in extra_info
        extra_data = extra_info[uuid]

        # Should have all columns but ID should not be duplicated
        expected_keys = {"id", "name", "guid", "station"}
        assert set(extra_data.keys()) == expected_keys

        # Verify values
        assert extra_data["id"] == expected_id
        assert extra_data["name"] == f"Node{expected_id}"
        assert extra_data["guid"] == f"g{expected_id}"
        assert extra_data["station"] == f"ST{expected_id}"


def test_normalize_extra_col_def_order_independence():
    """Test that _normalize_extra_col_def correctly handles GUID appearing in both optional_extra and regular extra"""
    # Arrange
    converter = TabularConverter(mapping_file=MAPPING_FILE)

    # This mimics the YAML config: optional_extra appears first with GUID and StationID,
    # then regular columns ID, Name, GUID follow
    col_def = [{"optional_extra": ["GUID", "StationID"]}, "ID", "Name", "GUID"]

    # Act
    result = converter._normalize_extra_col_def(col_def)

    # Assert
    # Expected: GUID should be removed from optional_extra since it appears as a regular column
    # Only StationID should remain in optional_extra
    # Regular columns ID, Name, GUID should be preserved in order
    expected = [{"optional_extra": ["StationID"]}, "ID", "Name", "GUID"]

    assert result == expected, f"Expected {expected}, got {result}"


def test_optional_extra_ordering_invariance_integration():
    """Integration test using vision_optional_extra_ordering_invariance.yaml to verify order independence"""
    # Arrange
    mapping_file = Path(__file__).parents[2] / "data" / "vision" / "vision_optional_extra_ordering_invariance.yaml"
    converter = TabularConverter(mapping_file=mapping_file)

    # Create test data with all columns: ID, Name, GUID, StationID
    data = TabularData(
        Nodes=pd.DataFrame(
            {
                "Number": [1, 2, 3],
                "Unom": [10.5, 10.5, 0.4],
                "ID": ["N001", "N002", "N003"],
                "Name": ["Node1", "Node2", "Node3"],
                "GUID": ["guid-1", "guid-2", "guid-3"],
                "StationID": ["ST1", "ST2", "ST3"],
            }
        )
    )

    extra_info: ExtraInfo = {}

    # Act
    result = converter._parse_data(data=data, data_type=DatasetType.input, extra_info=extra_info)

    # Assert
    assert ComponentType.node in result
    assert len(result[ComponentType.node]) == 3

    # Verify that extra_info contains the expected fields
    for node_id in result[ComponentType.node]["id"]:
        assert node_id in extra_info
        # Should have ID, Name, GUID (regular columns) and StationID (optional column)
        assert "ID" in extra_info[node_id]
        assert "Name" in extra_info[node_id]
        assert "GUID" in extra_info[node_id]
        assert "StationID" in extra_info[node_id]

        # GUID should appear only once (not duplicated from optional_extra)
        keys = list(extra_info[node_id].keys())
        assert keys.count("GUID") == 1, f"GUID should appear only once, but got: {keys}"

    # Verify values for first node
    first_node_id = result[ComponentType.node]["id"][0]
    assert extra_info[first_node_id]["ID"] == "N001"
    assert extra_info[first_node_id]["Name"] == "Node1"
    assert extra_info[first_node_id]["GUID"] == "guid-1"
    assert extra_info[first_node_id]["StationID"] == "ST1"
