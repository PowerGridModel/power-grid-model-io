# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import Callable
from unittest.mock import ANY, MagicMock, call, patch

import numpy as np
import pandapower as pp
import pandas as pd
import pytest
from power_grid_model import Branch3Side, BranchSide, LoadGenType, WindingType, initialize_array

from power_grid_model_io.converters.pandapower_converter import PandaPowerConverter

from ...utils import MockDf, MockFn, assert_struct_array_equal

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pandapower"


def _generate_ids(*args, **kwargs):
    return MockFn("_generate_ids", *args, **kwargs)


def _get_pgm_ids(*args, **kwargs):
    return MockFn("_get_pgm_ids", *args, **kwargs)


def _get_pp_attr(*args, **kwargs):
    return MockFn("_get_pp_attr", *args, **kwargs)


def get_switch_states(*args, **kwargs):
    return MockFn("get_switch_states", *args, **kwargs)


def get_trafo3w_switch_states(*args, **kwargs):
    return MockFn("get_trafo3w_switch_states", *args, **kwargs)


def get_trafo_winding_types(*args, **kwargs):
    return MockFn("get_trafo_winding_types", *args, **kwargs)


def get_trafo3w_winding_types(*args, **kwargs):
    return MockFn("get_trafo3w_winding_types", *args, **kwargs)


def _get_transformer_tap_side(*args, **kwargs):
    return MockFn("_get_transformer_tap_side", *args, **kwargs)


def _get_3wtransformer_tap_side(*args, **kwargs):
    return MockFn("_get_3wtransformer_tap_side", *args, **kwargs)


def _get_3wtransformer_tap_size(*args, **kwargs):
    return MockFn("_get_3wtransformer_tap_size", *args, **kwargs)


def _get_tap_size(*args, **kwargs):
    return MockFn("_get_tap_size", *args, **kwargs)


def np_array(*args, **kwargs):
    return MockFn("np.array", *args, **kwargs)

def _merge_to_pgm_data(*args, **kwargs):
    return MockFn("_merge_to_pgm_data", *args, **kwargs)

@pytest.fixture
def converter() -> PandaPowerConverter:
    converter = PandaPowerConverter()
    converter._generate_ids = MagicMock(side_effect=_generate_ids)  # type: ignore
    converter._get_pgm_ids = MagicMock(side_effect=_get_pgm_ids)  # type: ignore
    converter._get_pp_attr = MagicMock(side_effect=_get_pp_attr)  # type: ignore
    converter.get_switch_states = MagicMock(side_effect=get_switch_states)  # type: ignore
    converter.get_trafo_winding_types = MagicMock(side_effect=get_trafo_winding_types)  # type: ignore
    converter._get_tap_size = MagicMock(side_effect=_get_tap_size)  # type: ignore
    converter.get_trafo_winding_types = MagicMock(side_effect=get_trafo_winding_types)  # type: ignore
    converter.get_trafo3w_switch_states = MagicMock(side_effect=get_trafo3w_switch_states)  # type: ignore
    converter.get_trafo3w_winding_types = MagicMock(side_effect=get_trafo3w_winding_types)  # type: ignore
    converter._get_transformer_tap_side = MagicMock(side_effect=_get_transformer_tap_side)  # type: ignore
    converter._get_3wtransformer_tap_side = MagicMock(side_effect=_get_3wtransformer_tap_side)  # type: ignore
    converter._get_3wtransformer_tap_size = MagicMock(side_effect=_get_3wtransformer_tap_size)  # type: ignore
    converter._merge_to_pgm_data = MagicMock(side_effect=_merge_to_pgm_data)
    return converter


@pytest.fixture
def two_pp_objs() -> MockDf:
    return MockDf(2)


@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._fill_extra_info")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._create_input_data")
def test_parse_data(create_input_data_mock: MagicMock, fill_extra_info_mock: MagicMock):
    # Arrange
    converter = PandaPowerConverter()

    def create_input_data():
        converter.pgm_input_data = {"node": np.array([])}

    create_input_data_mock.side_effect = create_input_data

    # Act
    result = converter._parse_data(data={"bus": pd.DataFrame()}, data_type="input", extra_info=None)

    # Assert
    create_input_data_mock.assert_called_once_with()
    fill_extra_info_mock.assert_not_called()
    assert len(converter.pp_input_data) == 1 and "bus" in converter.pp_input_data
    assert len(converter.pgm_input_data) == 1 and "node" in converter.pgm_input_data
    assert len(result) == 1 and "node" in result


@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._fill_extra_info")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._create_input_data")
def test_parse_data__extra_info(create_input_data_mock: MagicMock, fill_extra_info_mock: MagicMock):
    # Arrange
    converter = PandaPowerConverter()

    extra_info = MagicMock("extra_info")

    # Act
    converter._parse_data(data={}, data_type="input", extra_info=extra_info)

    # Assert
    create_input_data_mock.assert_called_once_with()
    fill_extra_info_mock.assert_called_once_with(extra_info=extra_info)


def test_parse_data__update_data():
    # Arrange
    converter = PandaPowerConverter()

    # Act/Assert
    with pytest.raises(ValueError):
        converter._parse_data(data={}, data_type="update", extra_info=None)


def test_fill_extra_info():
    # Arrange
    converter = PandaPowerConverter()
    converter.idx_lookup[("bus", None)] = pd.Series([101, 102, 103], index=[0, 1, 2])
    converter.idx_lookup[("load", "const_current")] = pd.Series([201, 202, 203], index=[3, 4, 5])
    converter.pgm_input_data["sym_load"] = initialize_array("input", "sym_load", 3)
    converter.pgm_input_data["sym_load"]["id"] = [3, 4, 5]
    converter.pgm_input_data["sym_load"]["node"] = [0, 1, 2]
    converter.pgm_input_data["line"] = initialize_array("input", "line", 2)
    converter.pgm_input_data["line"]["id"] = [6, 7]
    converter.pgm_input_data["line"]["from_node"] = [0, 1]
    converter.pgm_input_data["line"]["to_node"] = [1, 2]

    # Act
    extra_info = {}
    converter._fill_extra_info(extra_info=extra_info)

    # Assert
    assert len(extra_info) == 8
    assert extra_info[0] == {"id_reference": {"table": "bus", "index": 101}}
    assert extra_info[1] == {"id_reference": {"table": "bus", "index": 102}}
    assert extra_info[2] == {"id_reference": {"table": "bus", "index": 103}}
    assert extra_info[3] == {"id_reference": {"table": "load", "name": "const_current", "index": 201}, "node": 0}
    assert extra_info[4] == {"id_reference": {"table": "load", "name": "const_current", "index": 202}, "node": 1}
    assert extra_info[5] == {"id_reference": {"table": "load", "name": "const_current", "index": 203}, "node": 2}
    assert extra_info[6] == {"from_node": 0, "to_node": 1}
    assert extra_info[7] == {"from_node": 1, "to_node": 2}


@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._extra_info_to_idx_lookup")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._extra_info_to_pgm_input_data")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._create_output_data")
def test__serialize_data(
    create_output_data_mock: MagicMock, extra_info_pgm_input_data: MagicMock, extra_info_to_idx_lookup: MagicMock
):
    # Arrange
    converter = PandaPowerConverter()

    def create_output_data():
        converter.pp_output_data = {"res_line": pd.DataFrame(np.array([]))}

    create_output_data_mock.side_effect = create_output_data

    # Act
    result = converter._serialize_data(data={"line": np.array([])}, extra_info=None)

    # Assert
    create_output_data_mock.assert_called_once_with()
    extra_info_to_idx_lookup.assert_not_called()
    extra_info_pgm_input_data.assert_not_called()
    assert len(converter.pp_output_data) == 1 and "res_line" in converter.pp_output_data
    assert len(converter.pgm_output_data) == 1 and "line" in converter.pgm_output_data
    assert len(result) == 1 and "res_line" in result


@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._extra_info_to_idx_lookup")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._extra_info_to_pgm_input_data")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._create_output_data")
def test_serialize_data__extra_info(
    create_output_data_mock: MagicMock,
    extra_info_pgm_input_data_mock: MagicMock,
    extra_info_to_idx_lookup_mock: MagicMock,
):
    # Arrange
    converter = PandaPowerConverter()

    extra_info = MagicMock("extra_info")

    # Act
    converter._serialize_data(data={}, extra_info=extra_info)

    # Assert
    create_output_data_mock.assert_called_once_with()
    extra_info_pgm_input_data_mock.assert_called_once_with(extra_info)
    extra_info_to_idx_lookup_mock.assert_called_once_with(extra_info)


def test_extra_info_to_idx_lookup():
    # Arrange
    converter = PandaPowerConverter()
    extra_info = {
        0: {"id_reference": {"table": "bus", "index": 101}},
        1: {"id_reference": {"table": "bus", "index": 102}},
        2: {"id_reference": {"table": "bus", "index": 103}},
        3: {"id_reference": {"table": "load", "name": "const_current", "index": 201}, "node": 0},
        4: {"id_reference": {"table": "load", "name": "const_current", "index": 202}, "node": 1},
        5: {"id_reference": {"table": "load", "name": "const_current", "index": 203}, "node": 2},
        6: {"from_node": 0, "to_node": 1},
        7: {"from_node": 1, "to_node": 2},
    }

    # Act
    converter._extra_info_to_idx_lookup(extra_info=extra_info)

    # Assert
    pd.testing.assert_series_equal(converter.idx[("bus", None)], pd.Series([0, 1, 2], index=[101, 102, 103]))
    pd.testing.assert_series_equal(converter.idx_lookup[("bus", None)], pd.Series([101, 102, 103], index=[0, 1, 2]))

    pd.testing.assert_series_equal(
        converter.idx[("load", "const_current")], pd.Series([3, 4, 5], index=[201, 202, 203])
    )
    pd.testing.assert_series_equal(
        converter.idx_lookup[("load", "const_current")], pd.Series([201, 202, 203], index=[3, 4, 5])
    )


def test_extra_info_to_pgm_input_data():
    # Arrange
    converter = PandaPowerConverter()
    converter.pgm_output_data["node"] = initialize_array("sym_output", "node", 3)
    converter.pgm_output_data["line"] = initialize_array("sym_output", "line", 2)
    converter.pgm_output_data["node"]["id"] = [1, 2, 3]
    converter.pgm_output_data["line"]["id"] = [12, 23]
    extra_info = {
        12: {"from_node": 1, "to_node": 2},
        23: {"from_node": 2, "to_node": 3},
    }

    # Act
    converter._extra_info_to_pgm_input_data(extra_info=extra_info)

    # Assert
    assert "node" not in converter.pgm_input_data
    assert_struct_array_equal(
        converter.pgm_input_data["line"],
        [{"id": 12, "from_node": 1, "to_node": 2}, {"id": 23, "from_node": 2, "to_node": 3}],
    )


def test_create_input_data():
    # Arrange
    converter = MagicMock()

    # Act
    PandaPowerConverter._create_input_data(self=converter)  # type: ignore

    # Assert
    assert len(converter.method_calls) == 16
    converter._create_pgm_input_nodes.assert_called_once_with()
    converter._create_pgm_input_lines.assert_called_once_with()
    converter._create_pgm_input_sources.assert_called_once_with()
    converter._create_pgm_input_sym_loads.assert_called_once_with()
    converter._create_pgm_input_asym_loads.assert_called_once_with()
    converter._create_pgm_input_shunts.assert_called_once_with()
    converter._create_pgm_input_transformers.assert_called_once_with()
    converter._create_pgm_input_sym_gens.assert_called_once_with()
    converter._create_pgm_input_asym_gens.assert_called_once_with()
    converter._create_pgm_input_three_winding_transformers.assert_called_once_with()
    converter._create_pgm_input_links.assert_called_once_with()
    converter._create_pgm_input_storages.assert_called_once_with()
    converter._create_pgm_input_impedances.assert_called_once_with()
    converter._create_pgm_input_wards.assert_called_once_with()
    converter._create_pgm_input_xwards.assert_called_once_with()
    converter._create_pgm_input_motors.assert_called_once_with()


def test_create_output_data():
    # Arrange
    converter = MagicMock()

    # Act
    PandaPowerConverter._create_output_data(self=converter)  # type: ignore

    # Assert
    assert len(converter.method_calls) == 8
    converter._pp_buses_output.assert_called_once_with()
    converter._pp_lines_output.assert_called_once_with()
    converter._pp_ext_grids_output.assert_called_once_with()
    converter._pp_loads_output.assert_called_once_with()
    converter._pp_shunts_output.assert_called_once_with()
    converter._pp_trafos_output.assert_called_once_with()
    converter._pp_sgens_output.assert_called_once_with()
    converter._pp_trafos3w_output.assert_called_once_with()


@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._pp_buses_output")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._pp_lines_output")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._pp_ext_grids_output")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._pp_loads_output")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._pp_shunts_output")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._pp_trafos_output")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._pp_sgens_output")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._pp_trafos3w_output")
def test_create_output_data_node_lookup(
    mock__pp_buses_output: MagicMock,
    mock__pp_lines_output: MagicMock,
    mock__pp_ext_grids_output: MagicMock,
    mock__pp_loads_output: MagicMock,
    mock__pp_shunts_output: MagicMock,
    mock__pp_trafos_output: MagicMock,
    mock__pp_sgens_output: MagicMock,
    mock__pp_trafos3w_output: MagicMock,
):
    # Arrange
    converter = PandaPowerConverter()
    converter.pgm_output_data = {
        "node": initialize_array("sym_output", "node", 3),
    }
    converter.pgm_output_data["node"]["id"] = [22, 32, 42]
    converter.pgm_output_data["node"]["u_pu"] = [31, 12, 4]
    converter.pgm_output_data["node"]["u_angle"] = [0.5, 1, 2]

    # Act
    converter._create_output_data()

    # Assert
    assert converter.pgm_nodes_lookup["u_pu"][22] == 31
    assert converter.pgm_nodes_lookup["u_pu"][32] == 12
    assert converter.pgm_nodes_lookup["u_pu"][42] == 4
    assert converter.pgm_nodes_lookup["u_degree"][22] == pytest.approx(28.6478897565)
    assert converter.pgm_nodes_lookup["u_degree"][32] == pytest.approx(57.2957795131)
    assert converter.pgm_nodes_lookup["u_degree"][42] == pytest.approx(114.591559026)


@pytest.mark.parametrize(
    ("create_fn", "table"),
    [
        (PandaPowerConverter._create_pgm_input_nodes, "bus"),
        (PandaPowerConverter._create_pgm_input_lines, "line"),
        (PandaPowerConverter._create_pgm_input_sources, "ext_grid"),
        (PandaPowerConverter._create_pgm_input_shunts, "shunt"),
        (PandaPowerConverter._create_pgm_input_sym_gens, "sgen"),
        (PandaPowerConverter._create_pgm_input_sym_loads, "load"),
        (PandaPowerConverter._create_pgm_input_transformers, "trafo"),
        (PandaPowerConverter._create_pgm_input_three_winding_transformers, "trafo3w"),
        (PandaPowerConverter._create_pgm_input_links, "switch"),
    ],
)
@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_object__empty(
    mock_init_array: MagicMock, create_fn: Callable[[PandaPowerConverter], None], table: str
):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data[table] = pd.DataFrame()  # type: ignore

    # Act
    create_fn(converter)

    # Assert
    mock_init_array.assert_not_called()


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_nodes(mock_init_array: MagicMock, two_pp_objs: MockDf, converter):
    # Arrange
    converter.pp_input_data["bus"] = two_pp_objs

    # Act
    converter._create_pgm_input_nodes()

    # Assert

    # administration
    converter._generate_ids.assert_called_once_with("bus", two_pp_objs.index)

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="node", shape=2)

    # retrieval
    converter._get_pp_attr.assert_any_call("bus", "vn_kv")
    assert len(converter._get_pp_attr.call_args_list) == 1

    # assignment
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("bus", two_pp_objs.index))
    pgm.assert_any_call("u_rated", _get_pp_attr("bus", "vn_kv") * 1e3)
    assert len(pgm.call_args_list) == 2

    # result
    assert converter.pgm_input_data["node"] == mock_init_array.return_value


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_lines(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["line"] = two_pp_objs

    # Act
    converter._create_pgm_input_lines()

    # Assert

    # administration
    converter.get_switch_states.assert_called_once_with("line")
    converter._generate_ids.assert_called_once_with("line", two_pp_objs.index)
    converter._get_pgm_ids.assert_any_call("bus", _get_pp_attr("line", "from_bus"))
    converter._get_pgm_ids.assert_any_call("bus", _get_pp_attr("line", "to_bus"))

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="line", shape=2)

    # retrieval
    converter._get_pp_attr.assert_any_call("line", "from_bus")
    converter._get_pp_attr.assert_any_call("line", "to_bus")
    converter._get_pp_attr.assert_any_call("line", "in_service", True)
    converter._get_pp_attr.assert_any_call("line", "length_km")
    converter._get_pp_attr.assert_any_call("line", "parallel", 1)
    converter._get_pp_attr.assert_any_call("line", "r_ohm_per_km")
    converter._get_pp_attr.assert_any_call("line", "x_ohm_per_km")
    converter._get_pp_attr.assert_any_call("line", "c_nf_per_km")
    converter._get_pp_attr.assert_any_call("line", "g_us_per_km", 0)
    converter._get_pp_attr.assert_any_call("line", "max_i_ka")
    converter._get_pp_attr.assert_any_call("line", "df", 1)
    assert len(converter._get_pp_attr.call_args_list) == 11

    # assignment
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("line", two_pp_objs.index))
    pgm.assert_any_call("from_node", _get_pgm_ids("bus", _get_pp_attr("line", "from_bus")))
    pgm.assert_any_call("from_status", _get_pp_attr("line", "in_service", True) & get_switch_states("line")["from"])
    pgm.assert_any_call("to_node", _get_pgm_ids("bus", _get_pp_attr("line", "to_bus")))
    pgm.assert_any_call("to_status", _get_pp_attr("line", "in_service", True) & get_switch_states("line")["to"])
    pgm.assert_any_call(
        "r1",
        _get_pp_attr("line", "r_ohm_per_km")
        * (_get_pp_attr("line", "length_km") / _get_pp_attr("line", "parallel", 1)),
    )
    pgm.assert_any_call(
        "x1",
        _get_pp_attr("line", "x_ohm_per_km")
        * (_get_pp_attr("line", "length_km") / _get_pp_attr("line", "parallel", 1)),
    )
    pgm.assert_any_call(
        "c1",
        _get_pp_attr("line", "c_nf_per_km")
        * _get_pp_attr("line", "length_km")
        * _get_pp_attr("line", "parallel", 1)
        * 1e-9,
    )
    pgm.assert_any_call(
        "tan1", _get_pp_attr("line", "g_us_per_km", 0) / _get_pp_attr("line", "c_nf_per_km") / (np.pi / 10)
    )
    pgm.assert_any_call(
        "i_n",
        _get_pp_attr("line", "max_i_ka") * 1e3 * _get_pp_attr("line", "df", 1) * _get_pp_attr("line", "parallel", 1),
    )
    assert len(pgm.call_args_list) == 10

    # result
    assert converter.pgm_input_data["line"] == mock_init_array.return_value


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_sources(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["ext_grid"] = two_pp_objs

    # Act
    converter._create_pgm_input_sources()

    # Assert

    # administration:
    converter._generate_ids.assert_called_once_with("ext_grid", two_pp_objs.index)

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="source", shape=2)

    # retrieval:
    converter._get_pp_attr.assert_any_call("ext_grid", "bus")
    converter._get_pp_attr.assert_any_call("ext_grid", "vm_pu", 1.0)
    converter._get_pp_attr.assert_any_call("ext_grid", "va_degree", 0.0)
    converter._get_pp_attr.assert_any_call("ext_grid", "s_sc_max_mva", np.nan)
    converter._get_pp_attr.assert_any_call("ext_grid", "rx_max", np.nan)
    converter._get_pp_attr.assert_any_call("ext_grid", "in_service", True)
    assert len(converter._get_pp_attr.call_args_list) == 6

    # assignment:
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("ext_grid", two_pp_objs.index))
    pgm.assert_any_call("node", _get_pgm_ids("bus", _get_pp_attr("ext_grid", "bus")))
    pgm.assert_any_call("status", _get_pp_attr("ext_grid", "in_service"))
    pgm.assert_any_call("u_ref", _get_pp_attr("ext_grid", "vm_pu", 1.0))
    pgm.assert_any_call("u_ref_angle", _get_pp_attr("ext_grid", "va_degree", 0.0) * (np.pi / 180))
    pgm.assert_any_call("sk", _get_pp_attr("ext_grid", "s_sc_max_mva", np.nan) * 1e6)
    pgm.assert_any_call("rx_ratio", _get_pp_attr("ext_grid", "rx_max", np.nan))
    assert len(pgm.call_args_list) == 7

    # result
    assert converter.pgm_input_data["source"] == mock_init_array.return_value


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_sym_loads(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["load"] = two_pp_objs
    pgm_attr = ["id", "node", "status", "p_specified", "q_specified", "type"]
    pgm = {attr: MagicMock() for attr in pgm_attr}
    mock_init_array.return_value = pgm
    slices = [slice(None, 2), slice(2, 4), slice(-2, None)]
    assert slices[0].indices(3 * 2) == (0, 2, 1)
    assert slices[1].indices(3 * 2) == (2, 4, 1)
    assert slices[2].indices(3 * 2) == (4, 6, 1)

    # Act
    converter._create_pgm_input_sym_loads()

    # Assert

    # administration:

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="sym_load", shape=3 * 2)

    # retrieval:
    converter._get_pp_attr.assert_any_call("load", "bus")
    converter._get_pp_attr.assert_any_call("load", "p_mw", 0.0)
    converter._get_pp_attr.assert_any_call("load", "q_mvar", 0.0)
    converter._get_pp_attr.assert_any_call("load", "const_z_percent", 0)
    converter._get_pp_attr.assert_any_call("load", "const_i_percent", 0)
    converter._get_pp_attr.assert_any_call("load", "scaling", 1)
    converter._get_pp_attr.assert_any_call("load", "in_service", True)
    # converter._get_pp_attr.assert_any_call("load", "type") # TODO add after asym conversion
    assert len(converter._get_pp_attr.call_args_list) == 7

    # assignment:
    for attr in pgm_attr:
        for s in slices:
            pgm[attr].__setitem__.assert_any_call(s, ANY)
        assert len(pgm[attr].__setitem__.call_args_list) == len(slices)

    # result
    assert converter.pgm_input_data["sym_load"] == pgm


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_asym_loads(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["asymmetric_load"] = two_pp_objs

    # Act
    converter._create_pgm_input_asym_loads()

    # Assert

    # administration:
    converter._generate_ids.assert_called_once_with("asymmetric_load", two_pp_objs.index)

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="asym_load", shape=2)

    # retrieval:
    converter._get_pp_attr.assert_any_call("asymmetric_load", "bus")
    converter._get_pp_attr.assert_any_call("asymmetric_load", "p_a_mw")
    converter._get_pp_attr.assert_any_call("asymmetric_load", "p_b_mw")
    converter._get_pp_attr.assert_any_call("asymmetric_load", "p_c_mw")
    converter._get_pp_attr.assert_any_call("asymmetric_load", "q_a_mvar")
    converter._get_pp_attr.assert_any_call("asymmetric_load", "q_b_mvar")
    converter._get_pp_attr.assert_any_call("asymmetric_load", "q_c_mvar")
    converter._get_pp_attr.assert_any_call("asymmetric_load", "scaling")
    converter._get_pp_attr.assert_any_call("asymmetric_load", "in_service")
    assert len(converter._get_pp_attr.call_args_list) == 9

    # assignment:
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("asymmetric_load", two_pp_objs.index))
    pgm.assert_any_call("node", _get_pgm_ids("bus", _get_pp_attr("asymmetric_load", "bus")))
    pgm.assert_any_call("status", _get_pp_attr("asymmetric_load", "in_service"))
    pgm.assert_any_call("p_specified", ANY)
    pgm.assert_any_call("q_specified", ANY)
    assert len(pgm.call_args_list) == 6
    # result
    assert converter.pgm_input_data["asym_load"] == mock_init_array.return_value


def test_create_pgm_input_transformers__tap_dependent_impedance():
    # Arrange
    pp_net: pp.pandapowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=0.0)
    args = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    pp.create_transformer_from_parameters(pp_net, *args, tap_dependent_impedance=True)

    converter = PandaPowerConverter()
    converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}

    # Act/Assert
    with pytest.raises(RuntimeError, match="not supported"):
        converter._create_pgm_input_transformers()


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_shunts(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["shunt"] = two_pp_objs

    # Act
    converter._create_pgm_input_shunts()

    # Assert

    # administration:
    converter._generate_ids.assert_called_once_with("shunt", two_pp_objs.index)

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="shunt", shape=2)

    # retrieval:
    converter._get_pp_attr.assert_any_call("shunt", "bus")
    converter._get_pp_attr.assert_any_call("shunt", "p_mw")
    converter._get_pp_attr.assert_any_call("shunt", "q_mvar")
    converter._get_pp_attr.assert_any_call("shunt", "vn_kv")
    converter._get_pp_attr.assert_any_call("shunt", "step", 1)
    converter._get_pp_attr.assert_any_call("shunt", "in_service", True)

    assert len(converter._get_pp_attr.call_args_list) == 6

    # assignment:
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("shunt", two_pp_objs.index))
    pgm.assert_any_call("node", _get_pgm_ids("bus", _get_pp_attr("shunt", "bus")))
    pgm.assert_any_call("status", _get_pp_attr("shunt", "in_service", True))
    pgm.assert_any_call(
        "g1",
        _get_pp_attr("shunt", "p_mw")
        * _get_pp_attr("shunt", "step", 1)
        / _get_pp_attr("shunt", "vn_kv")
        / _get_pp_attr("shunt", "vn_kv"),
    )
    pgm.assert_any_call(
        "b1",
        -_get_pp_attr("shunt", "q_mvar")
        * _get_pp_attr("shunt", "step", 1)
        / _get_pp_attr("shunt", "vn_kv")
        / _get_pp_attr("shunt", "vn_kv"),
    )

    assert len(pgm.call_args_list) == 5

    # result
    assert converter.pgm_input_data["shunt"] == mock_init_array.return_value


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
@patch("power_grid_model_io.converters.pandapower_converter.np.round", new=lambda x: x)
def test_create_pgm_input_transformers(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["trafo"] = two_pp_objs

    # Act
    converter._create_pgm_input_transformers()
    # Assert

    # administration:
    converter.get_switch_states.assert_called_once_with("trafo")
    converter._generate_ids.assert_called_once_with("trafo", two_pp_objs.index)
    converter._get_pgm_ids.assert_any_call("bus", _get_pp_attr("trafo", "hv_bus"))
    converter._get_pgm_ids.assert_any_call("bus", _get_pp_attr("trafo", "lv_bus"))
    converter._get_tap_size.assert_called_once_with(two_pp_objs)
    converter._get_transformer_tap_side.assert_called_once_with(_get_pp_attr("trafo", "tap_side"))

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="transformer", shape=2)

    # retrieval:
    converter._get_pp_attr.assert_any_call("trafo", "hv_bus")
    converter._get_pp_attr.assert_any_call("trafo", "lv_bus")
    converter._get_pp_attr.assert_any_call("trafo", "sn_mva")
    converter._get_pp_attr.assert_any_call("trafo", "vn_hv_kv")
    converter._get_pp_attr.assert_any_call("trafo", "vn_lv_kv")
    converter._get_pp_attr.assert_any_call("trafo", "vk_percent")
    converter._get_pp_attr.assert_any_call("trafo", "vkr_percent")
    converter._get_pp_attr.assert_any_call("trafo", "pfe_kw")
    converter._get_pp_attr.assert_any_call("trafo", "i0_percent")
    converter._get_pp_attr.assert_any_call("trafo", "shift_degree", 0.0)
    converter._get_pp_attr.assert_any_call("trafo", "tap_side")
    converter._get_pp_attr.assert_any_call("trafo", "tap_neutral", np.nan)
    converter._get_pp_attr.assert_any_call("trafo", "tap_min", np.nan)
    converter._get_pp_attr.assert_any_call("trafo", "tap_max", np.nan)
    converter._get_pp_attr.assert_any_call("trafo", "tap_pos", np.nan)
    converter._get_pp_attr.assert_any_call("trafo", "parallel", 1)
    converter._get_pp_attr.assert_any_call("trafo", "in_service", True)
    # converter._get_pp_attr.assert_any_call("trafo", "df")  #TODO add df in output conversions
    # converter._get_pp_attr.assert_any_call("trafo", "vk0_percent")  # TODO add checks after asym implementation
    # converter._get_pp_attr.assert_any_call("trafo", "vkr0_percent")  #
    # converter._get_pp_attr.assert_any_call("trafo", "mag0_percent")  #
    # converter._get_pp_attr.assert_any_call("trafo", "mag0_rx")  #
    # converter._get_pp_attr.assert_any_call("trafo", "si0_hv_partial")  #

    assert len(converter._get_pp_attr.call_args_list) == 17

    # assignment:
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("trafo", two_pp_objs.index))
    pgm.assert_any_call("from_node", _get_pgm_ids("bus", _get_pp_attr("trafo", "hv_bus")))
    pgm.assert_any_call("to_node", _get_pgm_ids("bus", _get_pp_attr("trafo", "lv_bus")))
    pgm.assert_any_call("from_status", ANY)
    pgm.assert_any_call("to_status", ANY)
    pgm.assert_any_call("u1", ANY)
    pgm.assert_any_call("u2", ANY)
    pgm.assert_any_call("sn", ANY)
    pgm.assert_any_call("uk", ANY)
    pgm.assert_any_call("pk", ANY)
    pgm.assert_any_call("i0", ANY)
    pgm.assert_any_call("p0", ANY)
    pgm.assert_any_call("winding_from", get_trafo_winding_types()["winding_from"])
    pgm.assert_any_call("winding_to", get_trafo_winding_types()["winding_to"])
    pgm.assert_any_call("clock", ANY)
    pgm.assert_any_call("tap_side", ANY)
    pgm.assert_any_call("tap_pos", ANY)
    pgm.assert_any_call("tap_min", ANY)
    pgm.assert_any_call("tap_max", ANY)
    pgm.assert_any_call("tap_nom", ANY)
    pgm.assert_any_call("tap_size", ANY)

    assert len(pgm.call_args_list) == 21

    # result
    assert converter.pgm_input_data["transformer"] == mock_init_array.return_value


@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter.get_switch_states",
    new=MagicMock(return_value=pd.DataFrame({"from": [True], "to": [True]})),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter.get_trafo_winding_types",
    new=MagicMock(return_value=pd.DataFrame({"winding_from": [0], "winding_to": [0]})),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._generate_ids",
    new=MagicMock(return_value=np.arange(1)),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._get_pgm_ids",
    new=MagicMock(return_value=pd.Series([0])),
)
def test_create_pgm_input_transformers__tap_side():
    # Arrange
    pp_net: pp.pandapowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=0.0)
    pp.create_transformer_from_parameters(
        pp_net, 0, 0, 0, 0, 0, 0, 0, 0, 0, tap_neutral=12.0, tap_pos=34.0, tap_side="hv"
    )
    pp.create_transformer_from_parameters(
        pp_net, 0, 0, 0, 0, 0, 0, 0, 0, 0, tap_neutral=12.0, tap_pos=34.0, tap_side="lv"
    )
    pp.create_transformer_from_parameters(
        pp_net, 0, 0, 0, 0, 0, 0, 0, 0, 0, tap_neutral=12.0, tap_pos=34.0, tap_side=None
    )

    converter = PandaPowerConverter()
    converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}

    # Act
    converter._create_pgm_input_transformers()
    result = converter.pgm_input_data["transformer"]

    # Assert
    assert result[0]["tap_side"] == BranchSide.from_side.value
    assert result[1]["tap_side"] == BranchSide.to_side.value
    assert result[2]["tap_side"] == BranchSide.from_side.value
    assert result[0]["tap_pos"] == 34.0 != result[0]["tap_nom"]
    assert result[1]["tap_pos"] == 34.0 != result[1]["tap_nom"]
    assert result[2]["tap_pos"] == 12.0 == result[2]["tap_nom"]


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_sym_gens(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["sgen"] = two_pp_objs

    # Act
    converter._create_pgm_input_sym_gens()

    # Assert

    # administration:
    converter._generate_ids.assert_called_once_with("sgen", two_pp_objs.index)

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="sym_gen", shape=2)

    # retrieval:
    converter._get_pp_attr.assert_any_call("sgen", "bus")
    converter._get_pp_attr.assert_any_call("sgen", "p_mw")
    converter._get_pp_attr.assert_any_call("sgen", "q_mvar", 0.0)
    converter._get_pp_attr.assert_any_call("sgen", "scaling", 1.0)
    converter._get_pp_attr.assert_any_call("sgen", "in_service", True)
    assert len(converter._get_pp_attr.call_args_list) == 5

    # assignment:
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("sgen", two_pp_objs.index))
    pgm.assert_any_call("node", _get_pgm_ids("bus", _get_pp_attr("sgen", "bus")))
    pgm.assert_any_call("status", _get_pp_attr("sgen", "in_service"))
    pgm.assert_any_call("type", LoadGenType.const_power)
    pgm.assert_any_call("p_specified", _get_pp_attr("sgen", "p_mw") * _get_pp_attr("sgen", "scaling", 1.0) * 1e6)
    pgm.assert_any_call("q_specified", _get_pp_attr("sgen", "q_mvar", 0.0) * _get_pp_attr("sgen", "scaling", 1.0) * 1e6)
    assert len(pgm.call_args_list) == 6

    # result
    assert converter.pgm_input_data["sym_gen"] == mock_init_array.return_value


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_asym_gens(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["asymmetric_sgen"] = two_pp_objs

    # Act
    converter._create_pgm_input_asym_gens()

    # Assert

    # administration:
    converter._generate_ids.assert_called_once_with("asymmetric_sgen", two_pp_objs.index)

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="asym_gen", shape=2)

    # retrieval:
    converter._get_pp_attr.assert_any_call("asymmetric_sgen", "bus")
    converter._get_pp_attr.assert_any_call("asymmetric_sgen", "p_a_mw")
    converter._get_pp_attr.assert_any_call("asymmetric_sgen", "p_b_mw")
    converter._get_pp_attr.assert_any_call("asymmetric_sgen", "p_c_mw")
    converter._get_pp_attr.assert_any_call("asymmetric_sgen", "q_a_mvar")
    converter._get_pp_attr.assert_any_call("asymmetric_sgen", "q_b_mvar")
    converter._get_pp_attr.assert_any_call("asymmetric_sgen", "q_c_mvar")
    converter._get_pp_attr.assert_any_call("asymmetric_sgen", "scaling")
    converter._get_pp_attr.assert_any_call("asymmetric_sgen", "in_service")
    assert len(converter._get_pp_attr.call_args_list) == 9

    # assignment:
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("asymmetric_sgen", two_pp_objs.index))
    pgm.assert_any_call("node", _get_pgm_ids("bus", _get_pp_attr("asymmetric_sgen", "bus")))
    pgm.assert_any_call("status", _get_pp_attr("asymmetric_sgen", "in_service"))
    pgm.assert_any_call("p_specified", ANY)
    pgm.assert_any_call("q_specified", ANY)
    pgm.assert_any_call("type", LoadGenType.const_power)
    assert len(pgm.call_args_list) == 6

    # result
    assert converter.pgm_input_data["asym_gen"] == mock_init_array.return_value


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
@patch("power_grid_model_io.converters.pandapower_converter.np.round", new=lambda x: x)
def test_create_pgm_input_three_winding_transformers(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["trafo3w"] = two_pp_objs

    # Act
    converter._create_pgm_input_three_winding_transformers()

    # Assert

    # administration:
    converter.get_trafo3w_switch_states.assert_called_once_with(two_pp_objs)
    converter._generate_ids.assert_called_once_with("trafo3w", two_pp_objs.index)
    converter._get_pgm_ids.assert_any_call("bus", _get_pp_attr("trafo3w", "hv_bus"))
    converter._get_pgm_ids.assert_any_call("bus", _get_pp_attr("trafo3w", "mv_bus"))
    converter._get_pgm_ids.assert_any_call("bus", _get_pp_attr("trafo3w", "lv_bus"))
    converter._get_3wtransformer_tap_size.assert_called_once_with(two_pp_objs)
    converter._get_3wtransformer_tap_side.assert_called_once_with(_get_pp_attr("trafo3w", "tap_side"))

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="three_winding_transformer", shape=2)

    # retrieval:
    converter._get_pp_attr.assert_any_call("trafo3w", "hv_bus")
    converter._get_pp_attr.assert_any_call("trafo3w", "mv_bus")
    converter._get_pp_attr.assert_any_call("trafo3w", "lv_bus")
    converter._get_pp_attr.assert_any_call("trafo3w", "vn_hv_kv")
    converter._get_pp_attr.assert_any_call("trafo3w", "vn_mv_kv")
    converter._get_pp_attr.assert_any_call("trafo3w", "vn_lv_kv")
    converter._get_pp_attr.assert_any_call("trafo3w", "sn_hv_mva")
    converter._get_pp_attr.assert_any_call("trafo3w", "sn_mv_mva")
    converter._get_pp_attr.assert_any_call("trafo3w", "sn_lv_mva")
    converter._get_pp_attr.assert_any_call("trafo3w", "vk_hv_percent")
    converter._get_pp_attr.assert_any_call("trafo3w", "vk_mv_percent")
    converter._get_pp_attr.assert_any_call("trafo3w", "vk_lv_percent")
    converter._get_pp_attr.assert_any_call("trafo3w", "vkr_hv_percent")
    converter._get_pp_attr.assert_any_call("trafo3w", "vkr_mv_percent")
    converter._get_pp_attr.assert_any_call("trafo3w", "vkr_lv_percent")
    converter._get_pp_attr.assert_any_call("trafo3w", "pfe_kw")
    converter._get_pp_attr.assert_any_call("trafo3w", "i0_percent")
    converter._get_pp_attr.assert_any_call("trafo3w", "shift_mv_degree", 0.0)
    converter._get_pp_attr.assert_any_call("trafo3w", "shift_lv_degree", 0.0)
    converter._get_pp_attr.assert_any_call("trafo3w", "tap_side")
    converter._get_pp_attr.assert_any_call("trafo3w", "tap_neutral", np.nan)
    converter._get_pp_attr.assert_any_call("trafo3w", "tap_min", np.nan)
    converter._get_pp_attr.assert_any_call("trafo3w", "tap_max", np.nan)
    converter._get_pp_attr.assert_any_call("trafo3w", "tap_pos", np.nan)
    converter._get_pp_attr.assert_any_call("trafo3w", "in_service", True)
    assert len(converter._get_pp_attr.call_args_list) == 25

    # assignment:
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("trafo3w", two_pp_objs.index))
    pgm.assert_any_call("node_1", _get_pgm_ids("bus", _get_pp_attr("trafo3w", "hv_bus")))
    pgm.assert_any_call("node_2", _get_pgm_ids("bus", _get_pp_attr("trafo3w", "mv_bus")))
    pgm.assert_any_call("node_3", _get_pgm_ids("bus", _get_pp_attr("trafo3w", "lv_bus")))
    pgm.assert_any_call("status_1", ANY)
    pgm.assert_any_call("status_2", ANY)
    pgm.assert_any_call("status_3", ANY)
    pgm.assert_any_call("u1", ANY)
    pgm.assert_any_call("u2", ANY)
    pgm.assert_any_call("u3", ANY)
    pgm.assert_any_call("sn_1", ANY)
    pgm.assert_any_call("sn_2", ANY)
    pgm.assert_any_call("sn_3", ANY)
    pgm.assert_any_call("uk_12", ANY)
    pgm.assert_any_call("uk_13", ANY)
    pgm.assert_any_call("uk_23", ANY)
    pgm.assert_any_call("pk_12", ANY)
    pgm.assert_any_call("pk_13", ANY)
    pgm.assert_any_call("pk_23", ANY)
    pgm.assert_any_call("i0", ANY)
    pgm.assert_any_call("p0", ANY)
    pgm.assert_any_call("winding_1", ANY)
    pgm.assert_any_call("winding_2", ANY)
    pgm.assert_any_call("winding_3", ANY)
    pgm.assert_any_call("clock_12", ANY)
    pgm.assert_any_call("clock_13", ANY)
    pgm.assert_any_call("tap_side", ANY)
    pgm.assert_any_call("tap_pos", ANY)
    pgm.assert_any_call("tap_min", ANY)
    pgm.assert_any_call("tap_max", ANY)
    pgm.assert_any_call("tap_nom", ANY)
    pgm.assert_any_call("tap_size", ANY)
    assert len(pgm.call_args_list) == 32

    # result
    assert converter.pgm_input_data["three_winding_transformer"] == mock_init_array.return_value


@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter.get_trafo3w_switch_states",
    new=MagicMock(return_value=pd.DataFrame({"side_1": [True], "side_2": [True], "side_3": [True]})),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter.get_trafo3w_winding_types",
    new=MagicMock(return_value=pd.DataFrame({"winding_1": [0], "winding_2": [0], "winding_3": [0]})),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._generate_ids",
    new=MagicMock(return_value=np.arange(1)),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._get_pgm_ids",
    new=MagicMock(return_value=pd.Series([0])),
)
def test_create_pgm_input_transformers3w__tap_side():
    # Arrange
    pp_net: pp.pandapowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=0.0)
    pp.create_transformer3w_from_parameters(
        pp_net, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, tap_neutral=12.0, tap_pos=34.0, tap_side="hv"
    )
    pp.create_transformer3w_from_parameters(
        pp_net, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, tap_neutral=12.0, tap_pos=34.0, tap_side="mv"
    )
    pp.create_transformer3w_from_parameters(
        pp_net, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, tap_neutral=12.0, tap_pos=34.0, tap_side="lv"
    )
    pp.create_transformer3w_from_parameters(
        pp_net, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, tap_neutral=12.0, tap_pos=34.0, tap_side=None
    )

    converter = PandaPowerConverter()
    converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}

    # Act
    converter._create_pgm_input_three_winding_transformers()
    result = converter.pgm_input_data["three_winding_transformer"]

    # Assert
    assert result[0]["tap_side"] == Branch3Side.side_1.value
    assert result[1]["tap_side"] == Branch3Side.side_2.value
    assert result[2]["tap_side"] == Branch3Side.side_3.value
    assert result[3]["tap_side"] == Branch3Side.side_1.value
    assert result[0]["tap_pos"] == 34.0 != result[0]["tap_nom"]
    assert result[1]["tap_pos"] == 34.0 != result[1]["tap_nom"]
    assert result[2]["tap_pos"] == 34.0 != result[2]["tap_nom"]
    assert result[3]["tap_pos"] == 12.0 == result[3]["tap_nom"]


def test_create_pgm_input_three_winding_transformers__tap_at_star_point():
    # Arrange
    pp_net: pp.pandapowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=0.0)
    args = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pp.create_transformer3w_from_parameters(pp_net, *args, tap_at_star_point=True)

    converter = PandaPowerConverter()
    converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}

    # Act/Assert
    with pytest.raises(RuntimeError, match="not supported"):
        converter._create_pgm_input_three_winding_transformers()


@pytest.mark.xfail(reason="https://github.com/e2nIEE/pandapower/issues/1831")
def test_create_pgm_input_three_winding_transformers__tap_dependent_impedance():
    # Arrange
    pp_net: pp.pandapowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=0.0)
    args = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pp.create_transformer3w_from_parameters(pp_net, *args, tap_dependent_impedance=True)

    converter = PandaPowerConverter()
    converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}

    # Act/Assert
    with pytest.raises(RuntimeError, match="not supported"):
        converter._create_pgm_input_three_winding_transformers()


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_links(mock_init_array: MagicMock, converter):
    # Arrange
    # three switches, of which two switches (#1 and #3) are bus to bus switches
    pp_switches = pd.DataFrame(
        [[0, 0, "b", False], [0, 0, "l", False], [0, 0, "b", False]],
        index=[1, 2, 3],
        columns=["bus", "element", "et", "closed"],
    )
    converter.pp_input_data["switch"] = pp_switches

    # Act
    converter._create_pgm_input_links()

    # Assert

    # administration:
    converter._generate_ids.assert_called_once_with("switch", ANY, name="bus_to_bus")
    pd.testing.assert_index_equal(converter._generate_ids.call_args_list[0].args[1], pd.Index([1, 3]))

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="link", shape=2)

    # retrieval:
    converter._get_pp_attr.assert_any_call("switch_b2b", "bus")
    converter._get_pp_attr.assert_any_call("switch_b2b", "element")
    converter._get_pp_attr.assert_any_call("switch_b2b", "closed", True)
    assert len(converter._get_pp_attr.call_args_list) == 3

    # assignment:
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", ANY)
    pgm.assert_any_call("from_node", ANY)
    pgm.assert_any_call("to_node", ANY)
    pgm.assert_any_call("from_status", ANY)
    pgm.assert_any_call("to_node", ANY)
    assert len(pgm.call_args_list) == 5

    # result
    assert converter.pgm_input_data["link"] == mock_init_array.return_value


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_storage(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["storage"] = two_pp_objs

    # Act / Assert
    converter._create_pgm_input_storages()

    # administration:
    converter._generate_ids.assert_called_once_with("storage", two_pp_objs.index, name="storage_load")

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="sym_load", shape=2)

    # retrieval:
    converter._get_pp_attr.assert_any_call("storage", "bus")
    converter._get_pp_attr.assert_any_call("storage", "p_mw")
    converter._get_pp_attr.assert_any_call("storage", "q_mvar")
    converter._get_pp_attr.assert_any_call("storage", "scaling")
    converter._get_pp_attr.assert_any_call("storage", "in_service")
    assert len(converter._get_pp_attr.call_args_list) == 5

    # assignment:
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", ANY)
    pgm.assert_any_call("node", ANY)
    pgm.assert_any_call("status", ANY)
    pgm.assert_any_call("p_specified", ANY)
    pgm.assert_any_call("q_specified", ANY)
    pgm.assert_any_call("type", ANY)
    assert len(pgm.call_args_list) == 6

    # result
    converter._merge_to_pgm_data.assert_called_once_with(pgm_name="sym_load", pgm_data_to_add=mock_init_array.return_value)


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_impedance(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["impedance"] = two_pp_objs

    # Act / Assert
    converter._create_pgm_input_impedances()

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="line", shape=len(two_pp_objs))

    # retrieval:
    converter._get_pp_attr.assert_any_call("impedance", "from_bus")
    converter._get_pp_attr.assert_any_call("impedance", "to_bus")
    converter._get_pp_attr.assert_any_call("impedance", "rft_pu")
    converter._get_pp_attr.assert_any_call("impedance", "xft_pu")
    converter._get_pp_attr.assert_any_call("impedance", "rtf_pu")
    converter._get_pp_attr.assert_any_call("impedance", "xtf_pu")
    converter._get_pp_attr.assert_any_call("impedance", "sn_mva")
    converter._get_pp_attr.assert_any_call("impedance", "in_service")
    assert len(converter._get_pp_attr.call_args_list) == 8

    # assignment
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("line", two_pp_objs.index))
    pgm.assert_any_call("from_node", _get_pgm_ids("bus", _get_pp_attr("impedance", "from_bus")))
    pgm.assert_any_call("from_status", _get_pp_attr("line", "in_service", True))
    pgm.assert_any_call("to_node", _get_pgm_ids("bus", _get_pp_attr("impedance", "to_bus")))
    pgm.assert_any_call("to_status", _get_pp_attr("line", "in_service", True))
    pgm.assert_any_call("r1", ANY)
    pgm.assert_any_call("x1", ANY)
    pgm.assert_any_call("c1", 0)
    pgm.assert_any_call("tan1", 0)
    pgm.assert_any_call("i_n", ANY)
    assert len(pgm.call_args_list) == 10
    # result
    assert converter.pgm_input_data["line"] == pgm


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_wards(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["ward"] = two_pp_objs
    pgm_attr = ["id", "node", "status", "p_specified", "q_specified", "type"]
    pgm = {attr: MagicMock() for attr in pgm_attr}
    mock_init_array.return_value = pgm
    slices = [slice(None, 2), slice(-2, None)]
    assert slices[0].indices(2 * 2) == (0, 2, 1)
    assert slices[1].indices(2 * 2) == (2, 4, 1)

    # Act
    converter._create_pgm_input_wards()

    # Assert

    # administration:

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="sym_load", shape=2 * 2)

    # retrieval:
    converter._get_pp_attr.assert_any_call("ward", "bus")
    converter._get_pp_attr.assert_any_call("ward", "ps_mw")
    converter._get_pp_attr.assert_any_call("ward", "qs_mvar")
    converter._get_pp_attr.assert_any_call("ward", "pz_mw")
    converter._get_pp_attr.assert_any_call("ward", "qz_mvar")
    converter._get_pp_attr.assert_any_call("ward", "in_service", True)
    assert len(converter._get_pp_attr.call_args_list) == 6

    # assignment:
    for attr in pgm_attr:
        for s in slices:
            pgm[attr].__setitem__.assert_any_call(s, ANY)
        assert len(pgm[attr].__setitem__.call_args_list) == len(slices)

    # result
    assert converter.pgm_input_data["sym_load"] == pgm


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_xward(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["xward"] = two_pp_objs
    converter.pp_input_data["bus"] = two_pp_objs
    pgm_attr = ["id", "node", "status", "p_specified", "q_specified", "type"]
    pgm = {attr: MagicMock() for attr in pgm_attr}
    mock_init_array.return_value = pgm
    slices = [slice(None, 2), slice(-2, None)]
    assert slices[0].indices(2 * 2) == (0, 2, 1)
    assert slices[1].indices(2 * 2) == (2, 4, 1)

    # Act
    converter._create_pgm_input_xwards()

    # Assert

    # administration:

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="sym_load", shape=2 * 2)

    # retrieval:
    converter._get_pp_attr.assert_any_call("xward", "bus")
    converter._get_pp_attr.assert_any_call("xward", "ps_mw")
    converter._get_pp_attr.assert_any_call("xward", "qs_mvar")
    converter._get_pp_attr.assert_any_call("xward", "pz_mw")
    converter._get_pp_attr.assert_any_call("xward", "qz_mvar")
    converter._get_pp_attr.assert_any_call("xward", "in_service", True)
    assert len(converter._get_pp_attr.call_args_list) == 6

    # assignment:
    for attr in pgm_attr:
        for s in slices:
            pgm[attr].__setitem__.assert_any_call(s, ANY)
        assert len(pgm[attr].__setitem__.call_args_list) == len(slices)

    # result
    assert converter.pgm_input_data["sym_load"] == pgm


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_motors(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["motor"] = two_pp_objs

    # Act
    converter._create_pgm_input_motors()

    # Assert

    # administration:
    converter._generate_ids.assert_called_once_with("motor", two_pp_objs.index, name="motor_load")

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="sym_load", shape=2)

    # retrieval:
    converter._get_pp_attr.assert_any_call("motor", "bus")
    converter._get_pp_attr.assert_any_call("motor", "pn_mech_mw")
    converter._get_pp_attr.assert_any_call("motor", "cos_phi")
    converter._get_pp_attr.assert_any_call("motor", "efficiency_percent")
    converter._get_pp_attr.assert_any_call("motor", "loading_percent")
    converter._get_pp_attr.assert_any_call("motor", "scaling")
    converter._get_pp_attr.assert_any_call("motor", "in_service")
    assert len(converter._get_pp_attr.call_args_list) == 7

    # assignment:
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", ANY)
    pgm.assert_any_call("node", ANY)
    pgm.assert_any_call("status", ANY)
    pgm.assert_any_call("p_specified", ANY)
    pgm.assert_any_call("q_specified", ANY)
    pgm.assert_any_call("type", ANY)
    assert len(pgm.call_args_list) == 6

    # result
    assert converter.pgm_input_data["sym_load"] == mock_init_array.return_value


def test_get_pgm_ids():
    # Arrange
    converter = PandaPowerConverter()
    converter.idx = {
        ("bus", None): pd.Series([10, 11, 12], index=[0, 1, 2]),
        ("load", "const_current"): pd.Series([13, 14], index=[3, 4]),
    }

    # Act
    bus_ids = converter._get_pgm_ids(pp_table="bus", pp_idx=pd.Series([2, 1]))
    load_ids = converter._get_pgm_ids(pp_table="load", name="const_current", pp_idx=pd.Series([3]))
    all_bus_ids = converter._get_pgm_ids(pp_table="bus")

    # Assert
    pd.testing.assert_series_equal(bus_ids, pd.Series([12, 11], index=[2, 1]))
    pd.testing.assert_series_equal(load_ids, pd.Series([13], index=[3]))
    pd.testing.assert_series_equal(all_bus_ids, pd.Series([10, 11, 12], index=[0, 1, 2]))


def test_get_pgm_ids__key_error():
    # Arrange
    converter = PandaPowerConverter()

    # Act / Assert
    with pytest.raises(KeyError, match=r"index.*bus"):
        converter._get_pgm_ids(pp_table="bus")


def test_get_pp_ids():
    # Arrange
    converter = PandaPowerConverter()
    converter.idx_lookup = {
        ("bus", None): pd.Series([0, 1, 2], index=[10, 11, 12]),
        ("load", "const_current"): pd.Series([3, 4], index=[13, 14]),
    }

    # Act
    bus_ids = converter._get_pp_ids(pp_table="bus", pgm_idx=pd.Series([12, 11]))
    load_ids = converter._get_pp_ids(pp_table="load", name="const_current", pgm_idx=pd.Series([13]))
    all_bus_ids = converter._get_pp_ids(pp_table="bus")

    # Assert
    pd.testing.assert_series_equal(bus_ids, pd.Series([2, 1], index=[12, 11]))
    pd.testing.assert_series_equal(load_ids, pd.Series([3], index=[13]))
    pd.testing.assert_series_equal(all_bus_ids, pd.Series([0, 1, 2], index=[10, 11, 12]))


def test_get_pp_ids__key_error():
    # Arrange
    converter = PandaPowerConverter()

    # Act / Assert
    with pytest.raises(KeyError, match=r"index.*bus"):
        converter._get_pp_ids(pp_table="bus")


def test_get_tap_size():
    # Arrange
    pp_trafo = pd.DataFrame(
        [["hv", 5.0, 10.5, 0.4], ["lv", 10.0, 10.5, 0.4], ["lv", np.nan, 10.5, 0.4]],
        columns=["tap_side", "tap_step_percent", "vn_hv_kv", "vn_lv_kv"],
    )
    expected_tap_size = np.array([525.0, 40.0, np.nan], dtype=np.float64)

    # Act
    actual_tap_size = PandaPowerConverter._get_tap_size(pp_trafo)

    # Assert
    np.testing.assert_array_equal(actual_tap_size, expected_tap_size)


def test_get_transformer_tap_side():
    # Arrange
    pp_trafo_tap_side = np.array(["hv", "lv", "lv", "hv"])
    expected_tap_side = np.array([0, 1, 1, 0], dtype=np.int8)

    # Act
    actual_tap_side = PandaPowerConverter._get_transformer_tap_side(pp_trafo_tap_side)

    # Assert
    np.testing.assert_array_equal(actual_tap_side, expected_tap_side)


def test_get_3wtransformer_tap_side():
    # Arrange
    pp_trafo3w_tap_side = np.array(["hv", "mv", "lv", None, "mv", "lv", "hv", "lv", None])
    expected_tap_side = np.array([0, 1, 2, 0, 1, 2, 0, 2, 0], dtype=np.int8)

    # Act
    actual_tap_side = PandaPowerConverter._get_3wtransformer_tap_side(pp_trafo3w_tap_side)

    # Assert
    np.testing.assert_array_equal(actual_tap_side, expected_tap_side)


def test_get_3wtransformer_tap_size():
    # Arrange
    pp_trafo = pd.DataFrame(
        [["hv", 62.0, 10.5, 400.0, 32.1], ["lv", 62.0, 10.5, 400.0, 32.1], ["mv", 62.0, 10.5, 400.0, 32.1]],
        columns=["tap_side", "tap_step_percent", "vn_hv_kv", "vn_mv_kv", "vn_lv_kv"],
    )
    expected_tap_size = np.array([6510.0, 19902.0, 248000.0], dtype=np.float64)

    # Act
    actual_tap_size = PandaPowerConverter._get_3wtransformer_tap_size(pp_trafo)

    # Assert
    np.testing.assert_array_equal(actual_tap_size, expected_tap_size)


@patch("power_grid_model_io.converters.pandapower_converter.get_winding")
def test_get_trafo_winding_types__vector_group(mock_get_winding: MagicMock):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {
        "trafo": pd.DataFrame([(1, "Dyn"), (2, "YNd"), (3, "Dyn")], columns=["id", "vector_group"])
    }
    mock_get_winding.side_effect = [WindingType.delta, WindingType.wye_n, WindingType.wye_n, WindingType.delta]
    expected = pd.DataFrame([(2, 1), (1, 2), (2, 1)], columns=["winding_from", "winding_to"])

    # Act
    actual = converter.get_trafo_winding_types()

    # Assert
    pd.testing.assert_frame_equal(actual, expected)
    assert len(mock_get_winding.call_args_list) == 4
    assert mock_get_winding.call_args_list[0] == call("D")
    assert mock_get_winding.call_args_list[1] == call("yn")
    assert mock_get_winding.call_args_list[2] == call("YN")
    assert mock_get_winding.call_args_list[3] == call("d")


@patch("power_grid_model_io.converters.pandapower_converter.get_winding")
def test_get_trafo_winding_types__std_types(mock_get_winding: MagicMock):
    # Arrange
    std_types = {"trafo": {"std_trafo_1": {"vector_group": "YNd"}, "std_trafo_2": {"vector_group": "Dyn"}}}
    converter = PandaPowerConverter(std_types=std_types)
    converter.pp_input_data = {
        "trafo": pd.DataFrame([(1, "std_trafo_2"), (2, "std_trafo_1"), (3, "std_trafo_2")], columns=["id", "std_type"])
    }
    mock_get_winding.side_effect = [WindingType.delta, WindingType.wye_n, WindingType.wye_n, WindingType.delta]
    expected = pd.DataFrame([(2, 1), (1, 2), (2, 1)], columns=["winding_from", "winding_to"])

    # Act
    actual = converter.get_trafo_winding_types()

    # Assert
    pd.testing.assert_frame_equal(actual, expected)
    assert len(mock_get_winding.call_args_list) == 4
    assert mock_get_winding.call_args_list[0] == call("D")
    assert mock_get_winding.call_args_list[1] == call("yn")
    assert mock_get_winding.call_args_list[2] == call("YN")
    assert mock_get_winding.call_args_list[3] == call("d")


@patch("power_grid_model_io.converters.pandapower_converter.get_winding")
def test_get_trafo3w_winding_types__vector_group(mock_get_winding: MagicMock):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {
        "trafo3w": pd.DataFrame([(1, "Dynz"), (2, "YNdy"), (3, "Dyny")], columns=["id", "vector_group"])
    }
    mock_get_winding.side_effect = [
        WindingType.delta,
        WindingType.wye_n,
        WindingType.zigzag,
        WindingType.wye_n,
        WindingType.delta,
        WindingType.wye,
        WindingType.delta,
        WindingType.wye_n,
        WindingType.wye,
    ]

    expected = pd.DataFrame([[2, 1, 3], [1, 2, 0], [2, 1, 0]], columns=["winding_1", "winding_2", "winding_3"])

    # Act
    actual = converter.get_trafo3w_winding_types()

    # Assert
    pd.testing.assert_frame_equal(actual, expected)
    assert len(mock_get_winding.call_args_list) == 9
    assert mock_get_winding.call_args_list[0] == call("D")
    assert mock_get_winding.call_args_list[1] == call("yn")
    assert mock_get_winding.call_args_list[2] == call("z")
    assert mock_get_winding.call_args_list[3] == call("YN")
    assert mock_get_winding.call_args_list[4] == call("d")
    assert mock_get_winding.call_args_list[5] == call("y")
    assert mock_get_winding.call_args_list[6] == call("D")
    assert mock_get_winding.call_args_list[7] == call("yn")
    assert mock_get_winding.call_args_list[8] == call("y")


@patch("power_grid_model_io.converters.pandapower_converter.get_winding")
def test_get_trafo3w_winding_types__std_types(mock_get_winding: MagicMock):
    # Arrange
    std_types = {"trafo3w": {"std_trafo3w_1": {"vector_group": "Dynz"}, "std_trafo3w_2": {"vector_group": "YNdy"}}}
    converter = PandaPowerConverter(std_types=std_types)
    converter.pp_input_data = {
        "trafo3w": pd.DataFrame(
            [(1, "std_trafo3w_2"), (2, "std_trafo3w_1"), (3, "std_trafo3w_2")], columns=["id", "std_type"]
        )
    }
    mock_get_winding.side_effect = [
        WindingType.wye_n,
        WindingType.delta,
        WindingType.wye,
        WindingType.delta,
        WindingType.wye_n,
        WindingType.zigzag,
        WindingType.wye_n,
        WindingType.delta,
        WindingType.wye,
    ]
    expected = pd.DataFrame([[1, 2, 0], [2, 1, 3], [1, 2, 0]], columns=["winding_1", "winding_2", "winding_3"])

    # Act
    actual = converter.get_trafo3w_winding_types()

    # Assert
    pd.testing.assert_frame_equal(actual, expected)
    assert len(mock_get_winding.call_args_list) == 6
    assert mock_get_winding.call_args_list[0] == call("YN")
    assert mock_get_winding.call_args_list[1] == call("d")
    assert mock_get_winding.call_args_list[2] == call("y")
    assert mock_get_winding.call_args_list[3] == call("D")
    assert mock_get_winding.call_args_list[4] == call("yn")
    assert mock_get_winding.call_args_list[5] == call("z")


def test_get_winding_types__value_error():
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {"trafo": pd.DataFrame([(1, "ADyn")], columns=["id", "vector_group"])}

    # Act / Assert
    with pytest.raises(ValueError):
        converter.get_trafo_winding_types()


def test_get_trafo3w_winding_types__value_error():
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {"trafo3w": pd.DataFrame([(1, "ADyndrr")], columns=["id", "vector_group"])}

    # Act / Assert
    with pytest.raises(ValueError):
        converter.get_trafo3w_winding_types()


def test_get_individual_switch_states():
    # Arrange
    pp_trafo = pd.DataFrame(
        columns=["index", "hv_bus"],
        data=[[1, 101], [2, 102], [3, 103]],
    )
    pp_switches = pd.DataFrame(
        columns=["element", "bus", "closed"],
        data=[[1, 101, False], [2, 202, False], [3, 103, True]],
    )

    expected_state = pd.Series([False, True, True], dtype=bool)

    # Act
    actual_state = PandaPowerConverter.get_individual_switch_states(pp_trafo, pp_switches, "hv_bus")

    # Assert
    np.testing.assert_array_equal(actual_state, expected_state)


def test_get_id():
    # Arrange
    converter = PandaPowerConverter()
    converter.idx = {("line", None): pd.Series([21, 345, 0, 3, 15], index=[0, 1, 2, 3, 4])}

    # Act
    actual_id = converter.get_id("line", 1)

    # Assert
    np.testing.assert_array_equal(actual_id, 345)


@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter.get_individual_switch_states")
def test_get_switch_states_lines(mock_get_individual_switch_states: MagicMock):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {
        "line": pd.DataFrame(columns=["from_bus", "to_bus"], data=[[101, 102]], index=[1]),
        "switch": pd.DataFrame(
            columns=["bus", "et", "element", "closed"],
            data=[[101, "l", 1, False], [102, "x", 1, False]],
            index=[1001, 1002],
        ),
    }
    mock_get_individual_switch_states.side_effect = [
        pd.Series([False], index=[1], dtype=bool, name="closed"),
        pd.Series([True], index=[1], dtype=bool, name="closed"),
    ]
    expected = pd.DataFrame(columns=["from", "to"], index=[1], data=[[False, True]])

    # Act
    actual = converter.get_switch_states("line")

    # Assert
    pd.testing.assert_frame_equal(actual, expected)

    assert len(mock_get_individual_switch_states.call_args_list) == 2

    assert mock_get_individual_switch_states.call_args_list[0] == call(ANY, ANY, "from_bus")
    pd.testing.assert_frame_equal(
        mock_get_individual_switch_states.call_args_list[0].args[0],
        pd.DataFrame(columns=["index", "from_bus"], data=[[1, 101]], index=[1]),
    )
    pd.testing.assert_frame_equal(
        mock_get_individual_switch_states.call_args_list[0].args[1],
        pd.DataFrame(columns=["element", "bus", "closed"], data=[[1, 101, False]], index=[1001]),
    )

    assert mock_get_individual_switch_states.call_args_list[1] == call(ANY, ANY, "to_bus")
    pd.testing.assert_frame_equal(
        mock_get_individual_switch_states.call_args_list[1].args[0],
        pd.DataFrame(columns=["index", "to_bus"], data=[[1, 102]], index=[1]),
    )
    pd.testing.assert_frame_equal(
        mock_get_individual_switch_states.call_args_list[1].args[1],
        pd.DataFrame(columns=["element", "bus", "closed"], data=[[1, 101, False]], index=[1001]),
    )


@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter.get_individual_switch_states")
def test_get_switch_states_trafos(mock_get_individual_switch_states: MagicMock):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {
        "trafo": pd.DataFrame([[2, 32, 31]], columns=["index", "hv_bus", "lv_bus"]),
        "switch": pd.DataFrame(
            [[101, 32, "t", 2, True], [321, 31, "t", 2, False]],
            columns=["index", "bus", "et", "element", "closed"],
        ),
    }
    mock_get_individual_switch_states.side_effect = [
        pd.Series([False], index=[2], dtype=bool, name="closed"),
        pd.Series([True], index=[2], dtype=bool, name="closed"),
    ]

    expected = pd.DataFrame(columns=["from", "to"], index=[2], data=[[False, True]])

    # Act
    actual = converter.get_switch_states("trafo")

    # Assert
    pd.testing.assert_frame_equal(actual, expected)
    assert len(mock_get_individual_switch_states.call_args_list) == 2


def test_get_switch_states__exception():
    # Arrange
    converter = PandaPowerConverter()

    # Act / Assert
    with pytest.raises(KeyError, match=r"link"):
        converter.get_switch_states("link")


@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter.get_individual_switch_states")
def test_get_trafo3w_switch_states(mock_get_individual_switch_states: MagicMock):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {
        "trafo3w": pd.DataFrame([[2, 32, 31, 315]], columns=["index", "hv_bus", "mv_bus", "lv_bus"]),
        "switch": pd.DataFrame(
            [[101, 315, "t3", 2, False], [321, 32, "t3", 2, False]],
            columns=["index", "bus", "et", "element", "closed"],
        ),
    }
    mock_get_individual_switch_states.side_effect = [False, True, False]

    expected = pd.DataFrame(columns=["side_1", "side_2", "side_3"], data=[[False, True, False]])

    # Act
    actual = converter.get_trafo3w_switch_states(converter.pp_input_data["trafo3w"])

    # Assert
    pd.testing.assert_frame_equal(actual, expected)
    assert len(mock_get_individual_switch_states.call_args_list) == 3


def test_lookup_id():
    # Arrange
    converter = PandaPowerConverter()
    converter.idx_lookup = {
        ("line", None): pd.Series([0, 1, 2, 3, 4], index=[21, 345, 0, 3, 15]),
        ("load", "const_current"): pd.Series([5, 6, 7, 8, 9], index=[543, 14, 34, 48, 4]),
    }

    expected_line = {"table": "line", "index": 4}
    expected_load = {"table": "load", "name": "const_current", "index": 8}

    # Act
    actual_line = converter.lookup_id(15)
    actual_load = converter.lookup_id(48)

    # Assert
    np.testing.assert_array_equal(actual_line, expected_line)
    np.testing.assert_array_equal(actual_load, expected_load)


def test_lookup_id__value_error():
    # Arrange
    converter = PandaPowerConverter()
    converter.idx_lookup = {("line", None): pd.Series([0, 1, 2, 3, 4], index=[21, 345, 0, 3, 15])}

    # Act / Assert
    with pytest.raises(KeyError):
        converter.lookup_id(5)


def test_pp_buses_output__accumulate_power__zero():
    # Arrange
    converter = PandaPowerConverter()
    converter.idx = {"bus": pd.Series([0, 1, 2, 3], index=[101, 102, 103, 104], dtype=np.int32)}
    pp_buses = pd.DataFrame(np.empty((4, 2), np.float64), columns=["p_mw", "q_mvar"], index=[101, 102, 103, 104])

    # Act
    converter._pp_buses_output__accumulate_power(pp_buses)

    # Assert
    assert pp_buses["p_mw"][101] == 0.0
    assert pp_buses["p_mw"][102] == 0.0
    assert pp_buses["p_mw"][103] == 0.0
    assert pp_buses["p_mw"][104] == 0.0
    assert pp_buses["q_mvar"][101] == 0.0
    assert pp_buses["q_mvar"][102] == 0.0
    assert pp_buses["q_mvar"][103] == 0.0
    assert pp_buses["q_mvar"][104] == 0.0


def test_pp_buses_output__accumulate_power():
    # Arrange
    converter = PandaPowerConverter()
    converter.idx_lookup = {("bus", None): pd.Series([101, 102, 103, 104], index=[0, 1, 2, 3], dtype=np.int32)}
    pp_buses = pd.DataFrame(np.empty((4, 2), np.float64), columns=["p_mw", "q_mvar"], index=[101, 102, 103, 104])

    converter.pgm_input_data = {
        "line": initialize_array("input", "line", 3),
        "link": initialize_array("input", "link", 2),
        "transformer": initialize_array("input", "transformer", 2),
        "three_winding_transformer": initialize_array("input", "three_winding_transformer", 2),
    }
    converter.pgm_output_data = {
        "line": initialize_array("sym_output", "line", 3),
        "link": initialize_array("sym_output", "link", 2),
        "transformer": initialize_array("sym_output", "transformer", 2),
        "three_winding_transformer": initialize_array("sym_output", "three_winding_transformer", 2),
    }
    converter.pgm_input_data["line"]["from_node"] = [0, 1, 1]
    converter.pgm_input_data["line"]["to_node"] = [1, 2, 3]
    converter.pgm_input_data["link"]["from_node"] = [0, 1]
    converter.pgm_input_data["link"]["to_node"] = [1, 2]
    converter.pgm_input_data["transformer"]["from_node"] = [0, 1]
    converter.pgm_input_data["transformer"]["to_node"] = [1, 2]
    converter.pgm_input_data["three_winding_transformer"]["node_1"] = [0, 1]
    converter.pgm_input_data["three_winding_transformer"]["node_2"] = [1, 2]
    converter.pgm_input_data["three_winding_transformer"]["node_3"] = [2, 3]
    converter.pgm_output_data["line"]["p_from"] = [1.0, 2.0, 4.0]
    converter.pgm_output_data["line"]["q_from"] = [0.1, 0.2, 0.4]
    converter.pgm_output_data["line"]["p_to"] = [-1.0, -2.0, -4.0]
    converter.pgm_output_data["line"]["q_to"] = [-0.1, -0.2, -0.4]
    converter.pgm_output_data["link"]["p_from"] = [10.0, 20.0]
    converter.pgm_output_data["link"]["q_from"] = [0.01, 0.02]
    converter.pgm_output_data["link"]["p_to"] = [-10.0, -20.0]
    converter.pgm_output_data["link"]["q_to"] = [-0.01, -0.02]
    converter.pgm_output_data["transformer"]["p_from"] = [100.0, 200.0]
    converter.pgm_output_data["transformer"]["q_from"] = [0.001, 0.002]
    converter.pgm_output_data["transformer"]["p_to"] = [-100.0, -200.0]
    converter.pgm_output_data["transformer"]["q_to"] = [-0.001, -0.002]
    converter.pgm_output_data["three_winding_transformer"]["p_1"] = [1000.0, 10000.0]
    converter.pgm_output_data["three_winding_transformer"]["q_1"] = [0.0001, 0.00001]
    converter.pgm_output_data["three_winding_transformer"]["p_2"] = [2000.0, 20000.0]
    converter.pgm_output_data["three_winding_transformer"]["q_2"] = [0.0002, 0.00002]
    converter.pgm_output_data["three_winding_transformer"]["p_3"] = [4000.0, 40000.0]
    converter.pgm_output_data["three_winding_transformer"]["q_3"] = [0.0004, 0.00004]

    # Act
    converter._pp_buses_output__accumulate_power(pp_buses)

    # Assert
    assert pp_buses["p_mw"][101] * 1e6 == -1.0 - 10.0 - 100.0 - 1000.0
    assert pp_buses["p_mw"][102] * 1e6 == -2.0 - 4.0 + 1.0 - 20.0 + 10.0 - 200.0 + 100.0 - 10000.0 - 2000.0
    assert pp_buses["p_mw"][103] * 1e6 == 2.0 + 20.0 + 200.0 - 20000.0 - 4000.0
    assert pp_buses["p_mw"][104] * 1e6 == 4.0 - 40000.0
    assert pp_buses["q_mvar"][101] * 1e6 == -0.1 - 0.01 - 0.001 - 0.0001
    assert pp_buses["q_mvar"][102] * 1e6 == -0.2 - 0.4 + 0.1 - 0.02 + 0.01 - 0.002 + 0.001 - 0.00001 - 0.0002
    assert pp_buses["q_mvar"][103] * 1e6 == 0.2 + 0.02 + 0.002 - 0.00002 - 0.0004
    assert pp_buses["q_mvar"][104] * 1e6 == 0.4 - 0.00004


def test_get_pp_attr_attribute_exists():
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {
        "trafo3w": pd.DataFrame([[2, 32, 31, 315]], columns=["index", "hv_bus", "mv_bus", "lv_bus"])
    }
    expected = np.array(32)

    # Act
    actual = converter._get_pp_attr("trafo3w", "hv_bus")

    # Assert
    np.testing.assert_array_equal(actual, expected)


def test_get_pp_attr_attribute_doesnt_exist():
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {"trafo3w": pd.DataFrame([[2, 31, 315]], columns=["index", "mv_bus", "lv_bus"])}

    # Act / Assert
    with pytest.raises(KeyError):
        converter._get_pp_attr("trafo3w", "hv_bus")


def test_get_pp_attr_use_default():
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {"trafo3w": pd.DataFrame([[2, 31, 315]], columns=["index", "mv_bus", "lv_bus"])}
    expected = np.array(625)

    # Act
    actual = converter._get_pp_attr("trafo3w", "hv_bus", 625)

    # Assert
    np.testing.assert_array_equal(actual, expected)


def test_get_pp_attr_from_std():
    # Arrange
    converter = PandaPowerConverter()
    converter._std_types = {"trafo3w": {"std_trafo3w_1": {"hv_bus": 964}}}
    converter.pp_input_data = {
        "trafo3w": pd.DataFrame([[2, 31, 315, "std_trafo3w_1"]], columns=["index", "mv_bus", "lv_bus", "std_type"])
    }

    expected = np.array(964)

    # Act
    actual = converter._get_pp_attr("trafo3w", "hv_bus")

    # Assert
    np.testing.assert_array_equal(actual, expected)


def test_get_pp_attr_default_after_checking_std():
    # Arrange
    converter = PandaPowerConverter()
    converter._std_types = {"trafo3w": {"std_trafo3w_1": {"lv_bus": 23}}}
    converter.pp_input_data = {
        "trafo3w": pd.DataFrame([[2, 31, 315, "std_trafo3w_1"]], columns=["index", "mv_bus", "lv_bus", "std_type"])
    }

    expected = np.array(964)

    # Act
    actual = converter._get_pp_attr("trafo3w", "hv_bus", 964)

    # Assert
    np.testing.assert_array_equal(actual, expected)


def test_get_pp_attr_error_after_checking_std():
    # Arrange
    converter = PandaPowerConverter()
    converter._std_types = {"trafo3w": {"std_trafo3w_1": {"lv_bus": 23}}}
    converter.pp_input_data = {
        "trafo3w": pd.DataFrame([[2, 31, 315, "std_trafo3w_1"]], columns=["index", "mv_bus", "lv_bus", "std_type"])
    }

    # Act/Assert
    with pytest.raises(KeyError):
        converter._get_pp_attr("trafo3w", "hv_bus")


def test_pp_loads_output():
    # Arrange
    converter = PandaPowerConverter()
    converter.pgm_output_data["sym_load"] = initialize_array("sym_output", "sym_load", 6)
    converter.pgm_output_data["sym_load"]["id"] = [0, 1, 2, 3, 4, 5]
    converter.pgm_output_data["sym_load"]["p"] = [1e6, 2e6, 4e6, 8e6, 16e6, 32e6]
    converter.pgm_output_data["sym_load"]["q"] = [1e4, 2e4, 4e4, 8e4, 16e4, 32e4]
    converter.idx[("load", "const_power")] = pd.Series([2, 4], index=[101, 100])
    converter.idx[("load", "const_current")] = pd.Series([1, 3], index=[102, 100])
    converter.idx[("load", "const_impedance")] = pd.Series([0, 5], index=[101, 102])

    expected = pd.DataFrame(
        [[16.0 + 8.0, 0.16 + 0.08], [4.0 + 1.0, 0.04 + 0.01], [2.0 + 32.0, 0.02 + 0.32]],
        columns=["p_mw", "q_mvar"],
        index=[100, 101, 102],
    )

    # Act
    converter._pp_loads_output()

    # Assert
    pd.testing.assert_frame_equal(converter.pp_output_data["res_load"], expected)
