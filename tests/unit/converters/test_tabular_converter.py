# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import Dict, Optional, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from power_grid_model import initialize_array, power_grid_meta_data
from power_grid_model.data_types import SingleDataset

from power_grid_model_io.converters.tabular_converter import COL_REF_RE, NODE_REF_RE, TabularConverter
from power_grid_model_io.data_types import ExtraInfoLookup, TabularData
from power_grid_model_io.mappings.tabular_mapping import InstanceAttributes

MAPPING_FILE = Path(__file__).parent / "test_data/mapping.yaml"


def ref_cases():
    yield "OtherTable!ValueColumn[IdColumn=RefColumn]", (
        "OtherTable",
        "ValueColumn",
        None,
        None,
        "IdColumn",
        None,
        None,
        "RefColumn",
    )

    yield "OtherTable!ValueColumn[OtherTable!IdColumn=ThisTable!RefColumn]", (
        "OtherTable",
        "ValueColumn",
        "OtherTable!",
        "OtherTable",
        "IdColumn",
        "ThisTable!",
        "ThisTable",
        "RefColumn",
    )

    yield "OtherTable.ValueColumn[IdColumn=RefColumn]", None
    yield "ValueColumn[IdColumn=RefColumn]", None
    yield "OtherTable![IdColumn=RefColumn]", None


def test_node_ref_pattern__pos():
    assert NODE_REF_RE.fullmatch("node")
    assert NODE_REF_RE.fullmatch("from_node")
    assert NODE_REF_RE.fullmatch("to_node")
    assert NODE_REF_RE.fullmatch("node_1")
    assert NODE_REF_RE.fullmatch("node_2")
    assert NODE_REF_RE.fullmatch("node_3")


def test_node_ref_pattern__neg():
    assert not NODE_REF_RE.fullmatch("nodes")
    assert not NODE_REF_RE.fullmatch("anode")
    assert not NODE_REF_RE.fullmatch("immunodeficient")


@pytest.mark.parametrize("value,groups", ref_cases())
def test_col_ref_pattern(value: str, groups: Optional[Tuple[Optional[str]]]):
    match = COL_REF_RE.fullmatch(value)
    if groups is None:
        assert match is None
    else:
        assert match is not None
        assert match.groups() == groups


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
    lines = pd.DataFrame(data=[[1, 1], [2, 2]], columns=["id_number", "from_node_side"])
    tabular_data_no_units_no_substitutions = TabularData(nodes=nodes, lines=lines)
    return tabular_data_no_units_no_substitutions


def test_converter__set_mapping_file(converter: TabularConverter):
    with pytest.raises(ValueError, match="Mapping file should be a .yaml file, .txt provided."):
        converter.set_mapping_file(mapping_file=Path("dummy/path.txt"))

    dummy_path = Path(__file__).parent / "test_data/dummy_mapping.yaml"
    with pytest.raises(KeyError, match="Missing 'grid' mapping in mapping_file"):
        converter.set_mapping_file(mapping_file=dummy_path)

    converter.set_mapping_file(mapping_file=MAPPING_FILE)


def test_converter__parse_data(converter: TabularConverter, tabular_data: TabularData):
    data = MagicMock()
    converter._parse_data(data=data, data_type="dummy")
    data.set_unit_multipliers.assert_called_once()
    data.set_substitutions.assert_called_once()

    pgm_input_data = converter._parse_data(data=tabular_data, data_type="input")
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
    assert pgm_input_data["sym_load"].dtype == power_grid_meta_data["input"]["sym_load"]["dtype"]


def test_converter__convert_table_to_component(
    converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData
):
    # if table does not exist in data _convert_table_to_component should return None
    none_data = converter._convert_table_to_component(
        data=tabular_data_no_units_no_substitutions,
        data_type="input",
        table="some_random_table",
        component="node",
        attributes={"key": "value"},
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
        )
    # wrong data_type
    with pytest.raises(KeyError, match="Invalid component type 'node' or data type 'some_type'"):
        converter._convert_table_to_component(
            data=tabular_data_no_units_no_substitutions,
            data_type="some_type",
            table="nodes",
            component="node",
            attributes={"key": "value"},
        )
    # no 'id' in attributes
    with pytest.raises(KeyError, match="No mapping for the attribute 'id' for 'nodes'!"):
        converter._convert_table_to_component(
            data=tabular_data_no_units_no_substitutions,
            data_type="input",
            table="nodes",
            component="node",
            attributes={"key": "value"},
        )

    node_attributes: InstanceAttributes = {"id": "id_number", "u_rated": "u_nom"}
    pgm_node_data = converter._convert_table_to_component(
        data=tabular_data_no_units_no_substitutions,
        data_type="input",
        table="nodes",
        component="node",
        attributes=node_attributes,
    )
    assert pgm_node_data is not None
    assert len(pgm_node_data) == 2
    assert (pgm_node_data["id"] == [0, 1]).all()
    assert (pgm_node_data["u_rated"] == [10.5e3, 400]).all()


def test_converter__convert_col_def_to_attribute(
    converter: TabularConverter,
    tabular_data_no_units_no_substitutions: TabularData,
    pgm_node_empty: SingleDataset,
    pgm_line_empty: SingleDataset,
    pgm_power_sensor_empty: SingleDataset,
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
        )

    # test "id"
    converter._convert_col_def_to_attribute(
        data=tabular_data_no_units_no_substitutions,
        pgm_data=pgm_node_empty["node"],
        table="nodes",
        component="node",
        attr="id",
        col_def="id_number",
    )
    assert len(pgm_node_empty) == 1
    assert (pgm_node_empty["node"]["id"] == [0, 1]).all()

    # test attr ends with "node"
    converter._convert_col_def_to_attribute(
        data=tabular_data_no_units_no_substitutions,
        pgm_data=pgm_line_empty["line"],
        table="lines",
        component="line",
        attr="from_node",
        col_def="from_node_side",
    )
    assert len(pgm_line_empty) == 1
    assert (pgm_line_empty["line"]["from_node"] == [0, 1]).all()

    # test attr ends with "object"
    with pytest.raises(
        NotImplementedError, match="dummys are not supported, because of the 'measured_object' reference..."
    ):
        converter._convert_col_def_to_attribute(
            data=tabular_data_no_units_no_substitutions,
            pgm_data=pgm_power_sensor_empty["power_sensor"],
            table="dummy",
            component="dummy",
            attr="measured_object",
            col_def="dummy",
        )

    # test extra info
    converter._convert_col_def_to_attribute(
        data=tabular_data_no_units_no_substitutions,
        pgm_data=pgm_node_empty["node"],
        table="nodes",
        component="node",
        attr="extra",
        col_def="u_nom",
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
    )
    assert len(pgm_node_empty) == 1
    assert (pgm_node_empty["node"]["u_rated"] == [10500.0, 400.0]).all()


def test_converter__handle_column(converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData):
    attr_data = converter._handle_column(
        data=tabular_data_no_units_no_substitutions, table="nodes", component="node", attr="u_rated", col_def="u_nom"
    )
    assert isinstance(attr_data, pd.Series)
    assert len(attr_data) == 2
    assert (attr_data == pd.Series([10500.0, 400.0])).all()

    with pytest.raises(
        ValueError,
        match=r"DataFrame for node.u_rated should contain a single column "
        r"\(Index\(\['id_number', 'u_nom'\], dtype='object'\)\)",
    ):
        converter._handle_column(
            data=tabular_data_no_units_no_substitutions,
            table="nodes",
            component="node",
            attr="u_rated",
            col_def=["id_number", "u_nom"],
        )


def test_converter__handle_id_column(converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData):
    extra_info: ExtraInfoLookup = {}
    uuids = converter._handle_id_column(
        data=tabular_data_no_units_no_substitutions,
        table="nodes",
        component="node",
        col_def="id_number",
        extra_info=extra_info,
    )
    assert (tabular_data_no_units_no_substitutions["nodes"]["id_number"] == pd.Series([1, 2])).all()
    assert (uuids == pd.Series([0, 1])).all()
    assert extra_info == {0: {"table": "nodes", "id_number": 1}, 1: {"table": "nodes", "id_number": 2}}


def test_converter__handle_extra_info(converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData):
    uuids = np.array([0, 1])
    # possible to call function with extra_info = None
    converter._handle_extra_info(
        data=tabular_data_no_units_no_substitutions, table="nodes", col_def="u_nom", uuids=uuids, extra_info=None
    )
    # _handle_extra_info expects the ids to be in extra_info already (as keys)
    with pytest.raises(KeyError, match="0"):
        converter._handle_extra_info(
            data=tabular_data_no_units_no_substitutions, table="nodes", col_def="u_nom", uuids=uuids, extra_info={}
        )
    extra_info: ExtraInfoLookup = {0: {"some_value": "some_key"}, 1: {"some_value": "some_key"}}
    converter._handle_extra_info(
        data=tabular_data_no_units_no_substitutions, table="nodes", col_def="u_nom", uuids=uuids, extra_info=extra_info
    )
    assert extra_info == {
        0: {"some_value": "some_key", "u_nom": 10500.0},
        1: {"some_value": "some_key", "u_nom": 400.0},
    }


def test_converter__handle_node_ref_column(
    converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData
):
    attr_data = converter._handle_node_ref_column(
        data=tabular_data_no_units_no_substitutions, table="lines", col_def="from_node_side"
    )
    assert (attr_data == pd.Series([0, 1])).all()


def test_converter__merge_pgm_data(converter: TabularConverter):
    nodes_1 = initialize_array("input", "node", 2)
    nodes_1["id"] = [0, 1]

    nodes_2 = initialize_array("input", "node", 3)
    nodes_2["id"] = [2, 3, 4]
    data = {"node": [nodes_1, nodes_2]}

    merged = converter._merge_pgm_data(data)
    assert len(merged) == 1
    assert (merged["node"]["id"] == np.array([0, 1, 2, 3, 4])).all()


def test_converter__serialize_data(converter: TabularConverter, pgm_node_empty: SingleDataset):
    with pytest.raises(NotImplementedError, match=r"Extra info can not \(yet\) be stored for tabular data"):
        converter._serialize_data(data=pgm_node_empty, extra_info={})
    with pytest.raises(NotImplementedError, match=r"Batch data can not \(yet\) be stored for tabular data"):
        converter._serialize_data(data=[])  # type: ignore
    # TODO: serialize_data expects pgm Dataset, TabularData, expects pd.DataFrame
    # tabular_data = converter._serialize_data(data=pgm_node_empty)


def test_converter__id_lookup(converter: TabularConverter):
    a01 = converter._id_lookup(component="a", row=pd.Series([0, 1]))
    b0 = converter._id_lookup(component="b", row=pd.Series([0]))
    b0_ = converter._id_lookup(component="b", row=pd.Series([0]))
    a0 = converter._id_lookup(component="a", row=pd.Series([0]))
    assert a01 == 0
    assert b0 == 1
    assert b0_ == 1
    assert a0 == 2


def test_converter__parse_col_def(converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData):
    with pytest.raises(TypeError, match=r"Invalid column definition: \(\)"):
        converter._parse_col_def(data=tabular_data_no_units_no_substitutions, table="table_name", col_def=())

    # type(col_def) == int
    with patch(
        "power_grid_model_io.converters.tabular_converter.TabularConverter._parse_col_def_const"
    ) as mock_parse_col_def_const:
        converter._parse_col_def(data=tabular_data_no_units_no_substitutions, table="nodes", col_def=50)
        mock_parse_col_def_const.assert_called_once_with(
            data=tabular_data_no_units_no_substitutions, table="nodes", col_def=50
        )

    # type(col_def) == float
    with patch(
        "power_grid_model_io.converters.tabular_converter.TabularConverter._parse_col_def_const"
    ) as mock_parse_col_def_const:
        converter._parse_col_def(data=tabular_data_no_units_no_substitutions, table="nodes", col_def=4.0)
        mock_parse_col_def_const.assert_called_once_with(
            data=tabular_data_no_units_no_substitutions, table="nodes", col_def=4.0
        )

    # type(col_def) == str (regular expression)
    with patch(
        "power_grid_model_io.converters.tabular_converter.TabularConverter._parse_col_def_column_reference"
    ) as mock_parse_col_column_reference:
        converter._parse_col_def(
            data=tabular_data_no_units_no_substitutions,
            table="nodes",
            col_def="OtherTable!ValueColumn[IdColumn=RefColumn]",
        )
        mock_parse_col_column_reference.assert_called_once_with(
            data=tabular_data_no_units_no_substitutions,
            table="nodes",
            col_def="OtherTable!ValueColumn[IdColumn=RefColumn]",
        )

    # type(col_def) == str
    with patch(
        "power_grid_model_io.converters.tabular_converter.TabularConverter._parse_col_def_column_name"
    ) as mock_parse_col_def_column_name:
        converter._parse_col_def(data=tabular_data_no_units_no_substitutions, table="nodes", col_def="col_name")
        mock_parse_col_def_column_name.assert_called_once_with(
            data=tabular_data_no_units_no_substitutions, table="nodes", col_def="col_name"
        )

    # type(col_def) == dict
    with patch(
        "power_grid_model_io.converters.tabular_converter.TabularConverter._parse_col_def_function"
    ) as mock_parse_col_def_function:
        converter._parse_col_def(data=tabular_data_no_units_no_substitutions, table="nodes", col_def={})
        mock_parse_col_def_function.assert_called_once_with(
            data=tabular_data_no_units_no_substitutions, table="nodes", col_def={}
        )

    # type(col_def) == list
    with patch(
        "power_grid_model_io.converters.tabular_converter.TabularConverter._parse_col_def_composite"
    ) as mock_parse_col_def_composite:
        converter._parse_col_def(data=tabular_data_no_units_no_substitutions, table="nodes", col_def=[])
        mock_parse_col_def_composite.assert_called_once_with(
            data=tabular_data_no_units_no_substitutions, table="nodes", col_def=[]
        )


def test_converter__parse_col_def_const(
    converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData
):
    with pytest.raises(AssertionError):
        converter._parse_col_def_const(
            data=tabular_data_no_units_no_substitutions, table="nodes", col_def="str"  # type: ignore
        )

    # type(col_def) == int
    col_int = converter._parse_col_def_const(data=tabular_data_no_units_no_substitutions, table="nodes", col_def=50)
    assert_frame_equal(col_int, pd.DataFrame([50, 50]))

    # type(col_def) == float
    col_int = converter._parse_col_def_const(data=tabular_data_no_units_no_substitutions, table="nodes", col_def=3.0)
    assert_frame_equal(col_int, pd.DataFrame([3.0, 3.0]))


def test_converter__parse_col_def_column_name(
    converter: TabularConverter, tabular_data_no_units_no_substitutions: TabularData
):
    with pytest.raises(AssertionError):
        converter._parse_col_def_column_name(
            data=tabular_data_no_units_no_substitutions, table="nodes", col_def=1  # type: ignore
        )

    df_multiple_columns = converter._parse_col_def_column_name(
        data=tabular_data_no_units_no_substitutions, table="nodes", col_def="  wrong_column  | id_number  | u_nom  "
    )
    assert_frame_equal(df_multiple_columns, pd.DataFrame([1, 2], columns=["id_number"]))

    df_inf = converter._parse_col_def_column_name(
        data=tabular_data_no_units_no_substitutions, table="nodes", col_def="inf"
    )
    assert_frame_equal(df_inf, pd.DataFrame([np.inf, np.inf]))

    with pytest.raises(KeyError, match="Could not find column 'a' and 'b' and 'c' on sheet 'nodes'"):
        converter._parse_col_def_column_name(
            data=tabular_data_no_units_no_substitutions, table="nodes", col_def="  a  | b  | c  "
        )


def test_converter__parse_col_def_column_reference():
    # TODO
    pass


def test_converter__parse_col_def_function():
    # TODO
    pass


def test_converter__parse_col_def_composite():
    # TODO
    pass
