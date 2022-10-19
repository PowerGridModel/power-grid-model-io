# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import Optional, Tuple
from unittest.mock import MagicMock

import pandas as pd
import pytest
from power_grid_model import initialize_array
from power_grid_model.data_types import SingleDataset

from power_grid_model_io.converters.tabular_converter import COL_REF_RE, NODE_REF_RE, TabularConverter
from power_grid_model_io.data_types import TabularData
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
    tabular_data = TabularData(nodes=nodes, lines=lines)
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
    assert list(pgm_input_data.keys()) == ["node", "line"]
    assert len(pgm_input_data["node"]) == 2
    assert (pgm_input_data["node"]["id"] == [0, 1]).all()
    assert (pgm_input_data["node"]["u_rated"] == [10.5e3, 400]).all()

    assert len(pgm_input_data["line"]) == 2
    assert (pgm_input_data["line"]["id"] == [2, 3]).all()
    assert (pgm_input_data["line"]["from_node"] == [0, 1]).all()


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

    # TODO test "invalid literal"


def test_converter__handle_column():
    # TODO
    pass


def test_converter__handle_id_column():
    # TODO
    pass


def test_converter__handle_extra_info():
    # TODO
    pass


def test_converter__handle_node_ref_column():
    # TODO
    pass


def test_converter__merge_pgm_data():
    # TODO
    pass


def test_converter__serialize_data(converter: TabularConverter, pgm_node_empty: SingleDataset):
    with pytest.raises(NotImplementedError, match=r"Extra info can not \(yet\) be stored for tabular data"):
        converter._serialize_data(data=pgm_node_empty, extra_info={})
    with pytest.raises(NotImplementedError, match=r"Batch data can not \(yet\) be stored for tabular data"):
        converter._serialize_data(data=[])  # type: ignore
    # TODO: serialize_data expects pgm Dataset, TabularData, expects pd.DataFrame
    # tabular_data = converter._serialize_data(data=pgm_input_data)


def test_converter__id_lookup():
    # TODO
    pass


def test_converter__parse_col_def():
    # TODO
    pass


def test_converter__parse_col_def_const():
    # TODO
    pass


def test_converter__parse_col_def_column_name():
    # TODO
    pass


def test_converter__parse_col_def_column_reference():
    # TODO
    pass


def test_converter__parse_col_def_function():
    # TODO
    pass


def test_converter__parse_col_def_composite():
    # TODO
    pass
