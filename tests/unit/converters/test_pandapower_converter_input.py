# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Callable
from unittest.mock import ANY, MagicMock, call, patch

import numpy as np
import pandapower as pp
import pandas as pd
import pytest
from power_grid_model import Branch3Side, BranchSide, LoadGenType, WindingType, initialize_array

from power_grid_model_io.converters.pandapower_converter import PandaPowerConverter

from ...utils import MockDf, MockFn, assert_struct_array_equal


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


@pytest.fixture
def converter() -> PandaPowerConverter:
    converter = PandaPowerConverter()
    converter._generate_ids = MagicMock(side_effect=_generate_ids)  # type: ignore
    converter._get_pgm_ids = MagicMock(side_effect=_get_pgm_ids)  # type: ignore
    converter._get_pp_attr = MagicMock(side_effect=_get_pp_attr)  # type: ignore
    converter.get_switch_states = MagicMock(side_effect=get_switch_states)  # type: ignore
    converter.get_trafo_winding_types = MagicMock(side_effect=get_trafo_winding_types)  # type: ignore
    converter._get_tap_size = MagicMock(side_effect=_get_tap_size)  # type: ignore
    converter.get_trafo_winding_types = MagicMock(side_effect=get_trafo_winding_types)  # type: ignore # TODO check this
    converter.get_trafo3w_switch_states = MagicMock(side_effect=get_trafo3w_switch_states)  # type: ignore
    converter.get_trafo3w_winding_types = MagicMock(side_effect=get_trafo3w_winding_types)  # type: ignore
    converter._get_transformer_tap_side = MagicMock(side_effect=_get_transformer_tap_side)  # type: ignore
    converter._get_3wtransformer_tap_side = MagicMock(side_effect=_get_3wtransformer_tap_side)  # type: ignore
    converter._get_3wtransformer_tap_size = MagicMock(side_effect=_get_3wtransformer_tap_size)  # type: ignore

    return converter


@pytest.fixture
def two_pp_objs() -> MockDf:
    return MockDf(2)


@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._fill_pgm_extra_info")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._fill_pp_extra_info")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._create_input_data")
def test_parse_data(
    create_input_data_mock: MagicMock, fill_pp_extra_info_mock: MagicMock, fill_pgm_extra_info_mock: MagicMock
):
    # Arrange
    converter = PandaPowerConverter()

    def create_input_data():
        converter.pgm_input_data = {"node": np.array([])}

    create_input_data_mock.side_effect = create_input_data

    # Act
    result = converter._parse_data(data={"bus": pd.DataFrame()}, data_type="input", extra_info=None)

    # Assert
    create_input_data_mock.assert_called_once_with()
    fill_pgm_extra_info_mock.assert_not_called()
    fill_pp_extra_info_mock.assert_not_called()
    assert len(converter.pp_input_data) == 1 and "bus" in converter.pp_input_data
    assert len(converter.pgm_input_data) == 1 and "node" in converter.pgm_input_data
    assert len(result) == 1 and "node" in result


@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._fill_pgm_extra_info")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._fill_pp_extra_info")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._create_input_data")
def test_parse_data__extra_info(
    create_input_data_mock: MagicMock, fill_pp_extra_info_mock: MagicMock, fill_pgm_extra_info_mock: MagicMock
):
    # Arrange
    converter = PandaPowerConverter()

    extra_info = MagicMock("extra_info")

    # Act
    converter._parse_data(data={}, data_type="input", extra_info=extra_info)

    # Assert
    create_input_data_mock.assert_called_once_with()
    fill_pgm_extra_info_mock.assert_called_once_with(extra_info=extra_info)
    fill_pp_extra_info_mock.assert_called_once_with(extra_info=extra_info)


def test_parse_data__update_data():
    # Arrange
    converter = PandaPowerConverter()

    # Act/Assert
    with pytest.raises(ValueError):
        converter._parse_data(data={}, data_type="update", extra_info=None)


def test_fill_pgm_extra_info():
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
    converter.pgm_input_data["line"]["i_n"] = [106.0, 105.0]

    # Act
    extra_info = {}
    converter._fill_pgm_extra_info(extra_info=extra_info)

    # Assert
    assert len(extra_info) == 8
    assert extra_info[0] == {"id_reference": {"table": "bus", "index": 101}}
    assert extra_info[1] == {"id_reference": {"table": "bus", "index": 102}}
    assert extra_info[2] == {"id_reference": {"table": "bus", "index": 103}}
    assert extra_info[3] == {
        "id_reference": {"table": "load", "name": "const_current", "index": 201},
        "pgm_input": {"node": 0},
    }
    assert extra_info[4] == {
        "id_reference": {"table": "load", "name": "const_current", "index": 202},
        "pgm_input": {"node": 1},
    }
    assert extra_info[5] == {
        "id_reference": {"table": "load", "name": "const_current", "index": 203},
        "pgm_input": {"node": 2},
    }
    assert extra_info[6] == {"pgm_input": {"from_node": 0, "to_node": 1, "i_n": 106.0}}
    assert extra_info[7] == {"pgm_input": {"from_node": 1, "to_node": 2, "i_n": 105.0}}


def test_fill_pp_extra_info():
    # Arrange
    converter = PandaPowerConverter()
    converter.idx_lookup[("line", None)] = pd.Series([102, 103], index=[1, 2])
    converter.idx_lookup[("trafo", None)] = pd.Series([201, 202, 203], index=[3, 4, 5])
    converter.idx[("line", None)] = pd.Series([1, 2], index=[102, 103])
    converter.idx[("trafo", None)] = pd.Series([3, 4, 5], index=[201, 202, 203])

    converter.pp_input_data["trafo"] = pd.DataFrame(
        {"df": [0.1, 0.2, 0.3], "other": [0.1, 0.2, 0.3]}, index=[201, 202, 203]
    )
    converter.pp_input_data["line"] = pd.DataFrame([10, 11, 12], columns=["df"], index=[201, 202, 203])

    # Act
    extra_info = {}
    converter._fill_pp_extra_info(extra_info=extra_info)

    # Assert
    assert len(extra_info) == 3
    assert extra_info[3] == {"pp_input": {"df": 0.1}}
    assert extra_info[4] == {"pp_input": {"df": 0.2}}
    assert extra_info[5] == {"pp_input": {"df": 0.3}}


def test_fill_pp_extra_info__no_info():
    # Arrange
    converter = PandaPowerConverter()
    converter.idx_lookup[("trafo", None)] = pd.Series([201, 202, 203], index=[3, 4, 5])
    converter.idx[("trafo", None)] = pd.Series([3, 4, 5], index=[201, 202, 203])

    converter.pp_input_data["trafo"] = pd.DataFrame(
        {"col1": [0.1, 0.2, 0.3], "col2": [0.1, 0.2, 0.3]}, index=[201, 202, 203]
    )
    converter.pgm_input_data["transformer"] = initialize_array("input", "transformer", 3)
    converter.pgm_input_data["transformer"]["id"] = [3, 4, 5]
    # Act
    extra_info = {}
    converter._fill_pp_extra_info(extra_info=extra_info)

    # Assert
    assert extra_info == {}


@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._extra_info_to_idx_lookup")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._extra_info_to_pgm_input_data")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._create_output_data")
def test__serialize_data__sym(
    create_output_data_mock: MagicMock, extra_info_pgm_input_data: MagicMock, extra_info_to_idx_lookup: MagicMock
):
    # Arrange
    converter = PandaPowerConverter()
    line_sym_output_array = initialize_array("sym_output", "line", 1)

    def create_output_data():
        converter.pp_output_data = {"res_line": pd.DataFrame(np.array([]))}

    create_output_data_mock.side_effect = create_output_data

    # Act
    result = converter._serialize_data(data={"line": line_sym_output_array}, extra_info=None)

    # Assert
    create_output_data_mock.assert_called_once_with()
    extra_info_to_idx_lookup.assert_not_called()
    extra_info_pgm_input_data.assert_not_called()
    assert len(converter.pp_output_data) == 1 and "res_line" in converter.pp_output_data
    assert len(converter.pgm_output_data) == 1 and "line" in converter.pgm_output_data
    assert len(result) == 1 and "res_line" in result


@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._extra_info_to_idx_lookup")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._extra_info_to_pgm_input_data")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._create_output_data_3ph")
def test__serialize_data__asym(
    create_output_data_3ph_mock: MagicMock, extra_info_pgm_input_data: MagicMock, extra_info_to_idx_lookup: MagicMock
):
    # Arrange
    converter = PandaPowerConverter()
    line_asym_output_array = initialize_array("asym_output", "line", 1)

    def create_output_data_3ph():
        converter.pp_output_data = {"res_line_3ph": pd.DataFrame(np.array([]))}

    create_output_data_3ph_mock.side_effect = create_output_data_3ph

    # Act
    result = converter._serialize_data(data={"line": line_asym_output_array}, extra_info=None)

    # Assert
    create_output_data_3ph_mock.assert_called_once_with()
    extra_info_to_idx_lookup.assert_not_called()
    extra_info_pgm_input_data.assert_not_called()
    assert len(converter.pp_output_data) == 1 and "res_line_3ph" in converter.pp_output_data
    assert len(converter.pgm_output_data) == 1 and "line" in converter.pgm_output_data
    assert len(result) == 1 and "res_line_3ph" in result


def test__serialize_data__invalid_output():
    # Arrange
    converter = PandaPowerConverter()

    # Act
    with pytest.raises(
        TypeError,
        match="Invalid output data dictionary supplied.",
    ):
        converter._serialize_data(data={"line": np.array([])}, extra_info=None)


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
        12: {"pgm_input": {"from_node": 1, "to_node": 2, "i_n": 105.0}},
        23: {"pgm_input": {"from_node": 2, "to_node": 3, "i_n": 5.0}},
    }

    # Act
    converter._extra_info_to_pgm_input_data(extra_info=extra_info)

    # Assert
    assert "node" not in converter.pgm_input_data
    assert_struct_array_equal(
        converter.pgm_input_data["line"],
        [{"id": 12, "from_node": 1, "to_node": 2, "i_n": 105.0}, {"id": 23, "from_node": 2, "to_node": 3, "i_n": 5.0}],
    )


def test__extra_info_to_pp_input_data():
    converter = PandaPowerConverter()
    converter.pgm_output_data["transformer"] = initialize_array("sym_output", "transformer", 3)
    converter.pgm_output_data["transformer"]["id"] = [3, 4, 5]

    converter.idx_lookup[("trafo", None)] = pd.Series([201, 202, 203], index=[3, 4, 5])
    converter.idx[("trafo", None)] = pd.Series([3, 4, 5], index=[201, 202, 203])
    extra_info = {
        3: {"pp_input": {"df": 0.1}},
        4: {"pp_input": {"df": 0.2}},
        5: {"pp_input": {"df": 0.3}},
    }

    expected_trafo = pd.DataFrame({"df": [0.1, 0.2, 0.3]}, index=[201, 202, 203])

    converter._extra_info_to_pp_input_data(extra_info)

    pd.testing.assert_frame_equal(converter.pp_input_data["trafo"], expected_trafo)


def test__extra_info_to_pp_input_data__empty():
    converter = PandaPowerConverter()
    converter.pgm_output_data["line"] = initialize_array("sym_output", "line", 3)

    converter._extra_info_to_pp_input_data({})
    assert len(converter.pp_input_data) == 0


def test_create_input_data():
    # Arrange
    converter = MagicMock()

    # Act
    PandaPowerConverter._create_input_data(self=converter)  # type: ignore

    # Assert
    assert len(converter.method_calls) == 18
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
    converter._create_pgm_input_generators.assert_called_once_with()
    converter._create_pgm_input_dclines.assert_called_once_with()


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
    converter._get_pp_attr.assert_any_call("bus", "vn_kv", expected_type="f8")
    assert len(converter._get_pp_attr.call_args_list) == 1

    # assignment
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("bus", two_pp_objs.index))
    pgm.assert_any_call("u_rated", _get_pp_attr("bus", "vn_kv", expected_type="f8") * 1e3)
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
    converter._get_pgm_ids.assert_any_call("bus", _get_pp_attr("line", "from_bus", expected_type="u4"))
    converter._get_pgm_ids.assert_any_call("bus", _get_pp_attr("line", "to_bus", expected_type="u4"))

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="line", shape=2)

    # retrieval
    converter._get_pp_attr.assert_any_call("line", "from_bus", expected_type="u4")
    converter._get_pp_attr.assert_any_call("line", "to_bus", expected_type="u4")
    converter._get_pp_attr.assert_any_call("line", "in_service", expected_type="bool", default=True)
    converter._get_pp_attr.assert_any_call("line", "length_km", expected_type="f8")
    converter._get_pp_attr.assert_any_call("line", "parallel", expected_type="u4", default=1)
    converter._get_pp_attr.assert_any_call("line", "r_ohm_per_km", expected_type="f8")
    converter._get_pp_attr.assert_any_call("line", "x_ohm_per_km", expected_type="f8")
    converter._get_pp_attr.assert_any_call("line", "c_nf_per_km", expected_type="f8")
    converter._get_pp_attr.assert_any_call("line", "g_us_per_km", expected_type="f8", default=0)
    converter._get_pp_attr.assert_any_call("line", "max_i_ka", expected_type="f8", default=np.nan)
    converter._get_pp_attr.assert_any_call("line", "df", expected_type="f8", default=1)
    converter._get_pp_attr.assert_any_call("line", "r0_ohm_per_km", expected_type="f8", default=np.nan)
    converter._get_pp_attr.assert_any_call("line", "x0_ohm_per_km", expected_type="f8", default=np.nan)
    converter._get_pp_attr.assert_any_call("line", "c0_nf_per_km", expected_type="f8", default=np.nan)
    converter._get_pp_attr.assert_any_call("line", "g0_us_per_km", expected_type="f8", default=0)
    assert len(converter._get_pp_attr.call_args_list) == 15

    # assignment
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("line", two_pp_objs.index))
    pgm.assert_any_call("from_node", _get_pgm_ids("bus", _get_pp_attr("line", "from_bus", expected_type="u4")))
    pgm.assert_any_call(
        "from_status",
        _get_pp_attr("line", "in_service", expected_type="bool", default=True) & get_switch_states("line")["from"],
    )
    pgm.assert_any_call("to_node", _get_pgm_ids("bus", _get_pp_attr("line", "to_bus", expected_type="u4")))
    pgm.assert_any_call(
        "to_status",
        _get_pp_attr("line", "in_service", expected_type="bool", default=True) & get_switch_states("line")["to"],
    )
    pgm.assert_any_call(
        "r1",
        _get_pp_attr("line", "r_ohm_per_km", expected_type="f8")
        * (
            _get_pp_attr("line", "length_km", expected_type="f8")
            / _get_pp_attr("line", "parallel", expected_type="u4", default=1)
        ),
    )
    pgm.assert_any_call(
        "x1",
        _get_pp_attr("line", "x_ohm_per_km", expected_type="f8")
        * (
            _get_pp_attr("line", "length_km", expected_type="f8")
            / _get_pp_attr("line", "parallel", expected_type="u4", default=1)
        ),
    )
    pgm.assert_any_call(
        "c1",
        _get_pp_attr("line", "c_nf_per_km", expected_type="f8")
        * _get_pp_attr("line", "length_km", expected_type="f8")
        * _get_pp_attr("line", "parallel", expected_type="u4", default=1)
        * 1e-9,
    )
    pgm.assert_any_call(
        "tan1",
        _get_pp_attr("line", "g_us_per_km", expected_type="f8", default=0)
        / _get_pp_attr("line", "c_nf_per_km", expected_type="f8")
        / (np.pi / 10),
    )
    pgm.assert_any_call(
        "i_n",
        (_get_pp_attr("line", "max_i_ka", expected_type="f8", default=np.nan) * 1e3)
        * _get_pp_attr("line", "df", expected_type="f8", default=1)
        * _get_pp_attr("line", "parallel", expected_type="u4", default=1),
    )
    pgm.assert_any_call("r0", ANY)
    pgm.assert_any_call("x0", ANY)
    pgm.assert_any_call("c0", ANY)
    pgm.assert_any_call("tan0", ANY)
    assert len(pgm.call_args_list) == 14

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
    converter._get_pp_attr.assert_any_call("ext_grid", "bus", expected_type="u4")
    converter._get_pp_attr.assert_any_call("ext_grid", "vm_pu", expected_type="f8", default=1.0)
    converter._get_pp_attr.assert_any_call("ext_grid", "va_degree", expected_type="f8", default=0.0)
    converter._get_pp_attr.assert_any_call("ext_grid", "s_sc_max_mva", expected_type="f8", default=np.nan)
    converter._get_pp_attr.assert_any_call("ext_grid", "rx_max", expected_type="f8", default=np.nan)
    converter._get_pp_attr.assert_any_call("ext_grid", "in_service", expected_type="bool", default=True)
    converter._get_pp_attr.assert_any_call("ext_grid", "r0x0_max", expected_type="f8", default=np.nan)
    converter._get_pp_attr.assert_any_call("ext_grid", "x0x_max", expected_type="f8", default=np.nan)
    assert len(converter._get_pp_attr.call_args_list) == 8

    # assignment:
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("ext_grid", two_pp_objs.index))
    pgm.assert_any_call("node", _get_pgm_ids("bus", _get_pp_attr("ext_grid", "bus", expected_type="u4")))
    pgm.assert_any_call("status", _get_pp_attr("ext_grid", "in_service", expected_type="bool", default=True))
    pgm.assert_any_call("u_ref", _get_pp_attr("ext_grid", "vm_pu", expected_type="f8", default=1.0))
    pgm.assert_any_call(
        "u_ref_angle", _get_pp_attr("ext_grid", "va_degree", expected_type="f8", default=0.0) * (np.pi / 180)
    )
    pgm.assert_any_call("sk", _get_pp_attr("ext_grid", "s_sc_max_mva", expected_type="f8", default=np.nan) * 1e6)
    pgm.assert_any_call("rx_ratio", _get_pp_attr("ext_grid", "rx_max", expected_type="f8", default=np.nan))
    assert len(pgm.call_args_list) == 7

    # result
    assert converter.pgm_input_data["source"] == mock_init_array.return_value


@pytest.mark.parametrize("kwargs", [{"r0x0_max": 0.5, "rx_max": 4}, {"x0x_max": 0.6}])
def test_create_pgm_input_sources__zero_sequence(kwargs) -> None:
    pp_net: pp.pandapowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=1.0)
    pp.create_ext_grid(pp_net, 0, **kwargs)

    converter = PandaPowerConverter()
    converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}
    converter.idx = {("bus", None): pd.Series([0], index=[0])}

    with patch("power_grid_model_io.converters.pandapower_converter.logger") as mock_logger:
        converter._create_pgm_input_sources()
        mock_logger.warning.assert_called_once()


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
    converter._get_pp_attr.assert_any_call("load", "bus", expected_type="u4")
    converter._get_pp_attr.assert_any_call("load", "p_mw", expected_type="f8", default=0.0)
    converter._get_pp_attr.assert_any_call("load", "q_mvar", expected_type="f8", default=0.0)
    converter._get_pp_attr.assert_any_call("load", "const_z_percent", expected_type="f8", default=0)
    converter._get_pp_attr.assert_any_call("load", "const_i_percent", expected_type="f8", default=0)
    converter._get_pp_attr.assert_any_call("load", "scaling", expected_type="f8", default=1)
    converter._get_pp_attr.assert_any_call("load", "in_service", expected_type="bool", default=True)
    converter._get_pp_attr.assert_any_call("load", "type", expected_type="O", default=None)
    assert len(converter._get_pp_attr.call_args_list) == 8

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
    converter._get_pp_attr.assert_any_call("asymmetric_load", "bus", expected_type="u4")
    converter._get_pp_attr.assert_any_call("asymmetric_load", "p_a_mw", expected_type="f8")
    converter._get_pp_attr.assert_any_call("asymmetric_load", "p_b_mw", expected_type="f8")
    converter._get_pp_attr.assert_any_call("asymmetric_load", "p_c_mw", expected_type="f8")
    converter._get_pp_attr.assert_any_call("asymmetric_load", "q_a_mvar", expected_type="f8")
    converter._get_pp_attr.assert_any_call("asymmetric_load", "q_b_mvar", expected_type="f8")
    converter._get_pp_attr.assert_any_call("asymmetric_load", "q_c_mvar", expected_type="f8")
    converter._get_pp_attr.assert_any_call("asymmetric_load", "scaling", expected_type="f8")
    converter._get_pp_attr.assert_any_call("asymmetric_load", "in_service", expected_type="bool", default=True)
    converter._get_pp_attr.assert_any_call("asymmetric_load", "type", expected_type="O", default=None)
    assert len(converter._get_pp_attr.call_args_list) == 10

    # assignment:
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("asymmetric_load", two_pp_objs.index))
    pgm.assert_any_call("node", _get_pgm_ids("bus", _get_pp_attr("asymmetric_load", "bus", expected_type="u4")))
    pgm.assert_any_call("status", _get_pp_attr("asymmetric_load", "in_service", expected_type="bool", default=True))
    pgm.assert_any_call("p_specified", ANY)
    pgm.assert_any_call("q_specified", ANY)
    assert len(pgm.call_args_list) == 6
    # result
    assert converter.pgm_input_data["asym_load"] == mock_init_array.return_value


def test_create_pgm_input_sym_loads__delta() -> None:
    # Arrange
    pp_net: pp.pandapowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=0.0)
    pp.create_load(pp_net, 0, 0, type="delta")

    converter = PandaPowerConverter()
    converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}

    # Act/Assert
    with pytest.raises(
        NotImplementedError, match="Delta loads are not implemented, only wye loads are supported in PGM."
    ):
        converter._create_pgm_input_sym_loads()


def test_create_pgm_input_asym_loads__delta() -> None:
    # Arrange
    pp_net: pp.pandapowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=0.0)
    pp.create_asymmetric_load(pp_net, 0, type="delta")

    converter = PandaPowerConverter()
    converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}

    # Act/Assert
    with pytest.raises(
        NotImplementedError, match="Delta loads are not implemented, only wye loads are supported in PGM."
    ):
        converter._create_pgm_input_asym_loads()


def test_create_pgm_input_transformers__tap_dependent_impedance() -> None:
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
    converter._get_pp_attr.assert_any_call("shunt", "bus", expected_type="u4")
    converter._get_pp_attr.assert_any_call("shunt", "p_mw", expected_type="f8")
    converter._get_pp_attr.assert_any_call("shunt", "q_mvar", expected_type="f8")
    converter._get_pp_attr.assert_any_call("shunt", "vn_kv", expected_type="f8")
    converter._get_pp_attr.assert_any_call("shunt", "step", expected_type="u4", default=1)
    converter._get_pp_attr.assert_any_call("shunt", "in_service", expected_type="bool", default=True)

    assert len(converter._get_pp_attr.call_args_list) == 6

    # assignment:
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("shunt", two_pp_objs.index))
    pgm.assert_any_call("node", _get_pgm_ids("bus", _get_pp_attr("shunt", "bus", expected_type="u4")))
    pgm.assert_any_call("status", _get_pp_attr("shunt", "in_service", expected_type="bool", default=True))
    pgm.assert_any_call(
        "g1",
        _get_pp_attr("shunt", "p_mw", expected_type="f8")
        * _get_pp_attr("shunt", "step", expected_type="u4", default=1)
        / _get_pp_attr("shunt", "vn_kv", expected_type="f8")
        / _get_pp_attr("shunt", "vn_kv", expected_type="f8"),
    )
    pgm.assert_any_call(
        "b1",
        -_get_pp_attr("shunt", "q_mvar", expected_type="f8")
        * _get_pp_attr("shunt", "step", expected_type="u4", default=1)
        / _get_pp_attr("shunt", "vn_kv", expected_type="f8")
        / _get_pp_attr("shunt", "vn_kv", expected_type="f8"),
    )
    pgm.assert_any_call("g0", ANY)
    pgm.assert_any_call("b0", ANY)

    assert len(pgm.call_args_list) == 7

    # result
    assert converter.pgm_input_data["shunt"] == mock_init_array.return_value


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
@patch("power_grid_model_io.converters.pandapower_converter.np.round", new=lambda x: x)
@patch("power_grid_model_io.converters.pandapower_converter.np.divide", new=lambda x, _, **kwargs: x)
@patch("power_grid_model_io.converters.pandapower_converter.np.bitwise_and", new=lambda x, _: x)
@patch("power_grid_model_io.converters.pandapower_converter.np.logical_and", new=lambda x, _: x)
@patch("power_grid_model_io.converters.pandapower_converter.np.allclose", new=lambda x, _: x)
@patch("power_grid_model_io.converters.pandapower_converter.np.isnan", new=lambda x: x)
def test_create_pgm_input_transformers(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["trafo"] = two_pp_objs

    # Act
    converter._create_pgm_input_transformers()
    # Assert

    # administration:
    converter.get_switch_states.assert_called_once_with("trafo")
    converter._generate_ids.assert_called_once_with("trafo", two_pp_objs.index)
    converter._get_pgm_ids.assert_any_call("bus", _get_pp_attr("trafo", "hv_bus", expected_type="u4"))
    converter._get_pgm_ids.assert_any_call("bus", _get_pp_attr("trafo", "lv_bus", expected_type="u4"))
    converter._get_tap_size.assert_called_once_with(two_pp_objs)
    converter._get_transformer_tap_side.assert_called_once_with(
        _get_pp_attr("trafo", "tap_side", expected_type="O", default=None)
    )

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="transformer", shape=2)

    # retrieval:
    converter._get_pp_attr.assert_any_call("trafo", "hv_bus", expected_type="u4")
    converter._get_pp_attr.assert_any_call("trafo", "lv_bus", expected_type="u4")
    converter._get_pp_attr.assert_any_call("trafo", "sn_mva", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo", "vn_hv_kv", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo", "vn_lv_kv", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo", "vk_percent", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo", "vkr_percent", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo", "pfe_kw", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo", "i0_percent", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo", "shift_degree", expected_type="f8", default=0.0)
    converter._get_pp_attr.assert_any_call("trafo", "tap_side", expected_type="O", default=None)
    converter._get_pp_attr.assert_any_call("trafo", "tap_neutral", expected_type="f8", default=np.nan)
    converter._get_pp_attr.assert_any_call("trafo", "tap_min", expected_type="i4", default=0)
    converter._get_pp_attr.assert_any_call("trafo", "tap_max", expected_type="i4", default=0)
    converter._get_pp_attr.assert_any_call("trafo", "tap_pos", expected_type="f8", default=np.nan)
    converter._get_pp_attr.assert_any_call("trafo", "parallel", expected_type="u4", default=1)
    converter._get_pp_attr.assert_any_call("trafo", "in_service", expected_type="bool", default=True)
    converter._get_pp_attr.assert_any_call("trafo", "vk0_percent", expected_type="f8", default=np.nan)
    converter._get_pp_attr.assert_any_call("trafo", "vkr0_percent", expected_type="f8", default=np.nan)
    converter._get_pp_attr.assert_any_call("trafo", "mag0_percent", expected_type="f8", default=np.nan)
    converter._get_pp_attr.assert_any_call("trafo", "mag0_rx", expected_type="f8", default=np.nan)
    converter._get_pp_attr.assert_any_call("trafo", "si0_hv_partial", expected_type="f8", default=np.nan)
    # converter._get_pp_attr.assert_any_call('trafo', 'df', expected_type='f8')  #TODO add df in output conversions
    assert len(converter._get_pp_attr.call_args_list) == 22

    # assignment:
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("trafo", two_pp_objs.index))
    pgm.assert_any_call("from_node", _get_pgm_ids("bus", _get_pp_attr("trafo", "hv_bus", expected_type="u4")))
    pgm.assert_any_call("to_node", _get_pgm_ids("bus", _get_pp_attr("trafo", "lv_bus", expected_type="u4")))
    pgm.assert_any_call("from_status", ANY)
    pgm.assert_any_call("to_status", ANY)
    pgm.assert_any_call("u1", ANY)
    pgm.assert_any_call("u2", ANY)
    pgm.assert_any_call("sn", ANY)
    pgm.assert_any_call("uk", ANY)
    pgm.assert_any_call("pk", ANY)
    pgm.assert_any_call("i0", ANY)
    pgm.assert_any_call("p0", ANY)
    pgm.assert_any_call("winding_from", ANY)
    pgm.assert_any_call("winding_to", ANY)
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
    new=MagicMock(
        return_value=pd.DataFrame(
            {
                "winding_from": [0, 0, 0, 0, 0, 0, WindingType.delta, WindingType.wye_n, 0, 0],
                "winding_to": [0, 0, 0, 0, 0, 0, WindingType.wye_n, WindingType.wye_n, 0, 0],
            }
        )
    ),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._generate_ids",
    new=MagicMock(return_value=np.arange(1)),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._get_pgm_ids",
    new=MagicMock(return_value=pd.Series([0])),
)
def test_create_pgm_input_transformers__default() -> None:
    # Arrange
    pp_net: pp.pandapowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=0.0)
    args = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    pp.create_transformer_from_parameters(
        pp_net, *args, tap_neutral=12.0, tap_pos=34.0, tap_step_percent=1, tap_side="hv"
    )
    pp.create_transformer_from_parameters(
        pp_net, *args, tap_neutral=12.0, tap_pos=34.0, tap_step_percent=1, tap_side="lv"
    )
    pp.create_transformer_from_parameters(
        pp_net, *args, tap_neutral=12.0, tap_pos=34.0, tap_step_percent=1, tap_side=None
    )
    tap_pos_trafo = pp.create_transformer_from_parameters(pp_net, *args, tap_neutral=12.0, tap_size=1, tap_side="hv")
    pp_net["trafo"].loc[tap_pos_trafo, "tap_pos"] = np.nan
    pp.create_transformer_from_parameters(pp_net, *args, tap_neutral=np.nan, tap_pos=34.0, tap_side="hv")
    pp.create_transformer_from_parameters(
        pp_net, *args, tap_neutral=12, tap_step_percent=np.nan, tap_pos=34.0, tap_side="hv"
    )
    pp.create_transformer_from_parameters(pp_net, *args, vector_group=None, shift_degree=30)
    pp.create_transformer_from_parameters(pp_net, *args, vector_group=None, shift_degree=60)
    pp.create_transformer_from_parameters(pp_net, *args, vector_group=None, shift_degree=59)
    pp.create_transformer_from_parameters(pp_net, *args, vector_group=None, shift_degree=61)

    converter = PandaPowerConverter()
    converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}

    # Act
    converter._create_pgm_input_transformers()
    result = converter.pgm_input_data["transformer"]

    # Assert
    assert result[0]["tap_side"] == BranchSide.from_side.value
    assert result[1]["tap_side"] == BranchSide.to_side.value
    assert result[2]["tap_side"] == BranchSide.from_side.value
    assert result[3]["tap_side"] == BranchSide.from_side.value
    assert result[4]["tap_side"] == BranchSide.from_side.value
    assert result[0]["tap_pos"] == 34.0 != result[0]["tap_nom"]
    assert result[1]["tap_pos"] == 34.0 != result[1]["tap_nom"]
    assert result[2]["tap_pos"] == 0.0 == result[2]["tap_nom"]
    assert result[3]["tap_pos"] == 0.0 == result[3]["tap_nom"]
    assert result[4]["tap_pos"] == 0.0 == result[4]["tap_nom"]
    assert result[5]["tap_size"] == 0.0

    assert result[6]["winding_from"] == WindingType.delta
    assert result[6]["winding_to"] == WindingType.wye_n
    assert result[7]["winding_from"] == WindingType.wye_n
    assert result[7]["winding_to"] == WindingType.wye_n

    assert result[8]["clock"] == 2
    assert result[9]["clock"] == 2


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
    converter._get_pp_attr.assert_any_call("sgen", "bus", expected_type="i8")
    converter._get_pp_attr.assert_any_call("sgen", "p_mw", expected_type="f8")
    converter._get_pp_attr.assert_any_call("sgen", "q_mvar", expected_type="f8", default=0.0)
    converter._get_pp_attr.assert_any_call("sgen", "scaling", expected_type="f8", default=1.0)
    converter._get_pp_attr.assert_any_call("sgen", "in_service", expected_type="bool", default=True)
    assert len(converter._get_pp_attr.call_args_list) == 5

    # assignment:
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("sgen", two_pp_objs.index))
    pgm.assert_any_call("node", _get_pgm_ids("bus", _get_pp_attr("sgen", "bus", expected_type="i8")))
    pgm.assert_any_call("status", _get_pp_attr("sgen", "in_service", expected_type="bool", default=True))
    pgm.assert_any_call("type", LoadGenType.const_power)
    pgm.assert_any_call(
        "p_specified",
        _get_pp_attr("sgen", "p_mw", expected_type="f8")
        * _get_pp_attr("sgen", "scaling", expected_type="f8", default=1.0)
        * 1e6,
    )
    pgm.assert_any_call(
        "q_specified",
        _get_pp_attr("sgen", "q_mvar", expected_type="f8", default=0.0)
        * _get_pp_attr("sgen", "scaling", expected_type="f8", default=1.0)
        * 1e6,
    )
    assert len(pgm.call_args_list) == 6

    # result
    assert converter.pgm_input_data["sym_gen"] == mock_init_array.return_value


@pytest.mark.parametrize(
    "kwargs",
    [{"vk0_percent": 2}, {"vkr0_percent": 1}, {"mag0_percent": 5}, {"mag0_rx": 0.2}, {"si0_hv_partial": 0.3}],
)
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
def test_create_pgm_input_transformers__zero_sequence(kwargs) -> None:
    # Arrange
    pp_net: pp.pandapowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=0.0)
    args = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    pp.create_transformer_from_parameters(pp_net, *args, **kwargs)

    converter = PandaPowerConverter()
    converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}

    with patch("power_grid_model_io.converters.pandapower_converter.logger") as mock_logger:
        converter._create_pgm_input_transformers()
        mock_logger.warning.assert_called_once()


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
    converter._get_pp_attr.assert_any_call("asymmetric_sgen", "bus", expected_type="i8")
    converter._get_pp_attr.assert_any_call("asymmetric_sgen", "p_a_mw", expected_type="f8")
    converter._get_pp_attr.assert_any_call("asymmetric_sgen", "p_b_mw", expected_type="f8")
    converter._get_pp_attr.assert_any_call("asymmetric_sgen", "p_c_mw", expected_type="f8")
    converter._get_pp_attr.assert_any_call("asymmetric_sgen", "q_a_mvar", expected_type="f8")
    converter._get_pp_attr.assert_any_call("asymmetric_sgen", "q_b_mvar", expected_type="f8")
    converter._get_pp_attr.assert_any_call("asymmetric_sgen", "q_c_mvar", expected_type="f8")
    converter._get_pp_attr.assert_any_call("asymmetric_sgen", "scaling", expected_type="f8")
    converter._get_pp_attr.assert_any_call("asymmetric_sgen", "in_service", expected_type="bool", default=True)
    assert len(converter._get_pp_attr.call_args_list) == 9

    # assignment:
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("asymmetric_sgen", two_pp_objs.index))
    pgm.assert_any_call("node", _get_pgm_ids("bus", _get_pp_attr("asymmetric_sgen", "bus", expected_type="i8")))
    pgm.assert_any_call("status", _get_pp_attr("asymmetric_sgen", "in_service", expected_type="bool", default=True))
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
    converter._get_pgm_ids.assert_any_call("bus", _get_pp_attr("trafo3w", "hv_bus", expected_type="u4"))
    converter._get_pgm_ids.assert_any_call("bus", _get_pp_attr("trafo3w", "mv_bus", expected_type="u4"))
    converter._get_pgm_ids.assert_any_call("bus", _get_pp_attr("trafo3w", "lv_bus", expected_type="u4"))
    converter._get_3wtransformer_tap_size.assert_called_once_with(two_pp_objs)
    converter._get_3wtransformer_tap_side.assert_called_once_with(
        _get_pp_attr("trafo3w", "tap_side", expected_type="O", default=None)
    )

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="three_winding_transformer", shape=2)

    # retrieval:
    converter._get_pp_attr.assert_any_call("trafo3w", "hv_bus", expected_type="u4")
    converter._get_pp_attr.assert_any_call("trafo3w", "mv_bus", expected_type="u4")
    converter._get_pp_attr.assert_any_call("trafo3w", "lv_bus", expected_type="u4")
    converter._get_pp_attr.assert_any_call("trafo3w", "vn_hv_kv", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo3w", "vn_mv_kv", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo3w", "vn_lv_kv", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo3w", "sn_hv_mva", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo3w", "sn_mv_mva", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo3w", "sn_lv_mva", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo3w", "vk_hv_percent", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo3w", "vk_mv_percent", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo3w", "vk_lv_percent", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo3w", "vkr_hv_percent", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo3w", "vkr_mv_percent", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo3w", "vkr_lv_percent", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo3w", "pfe_kw", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo3w", "i0_percent", expected_type="f8")
    converter._get_pp_attr.assert_any_call("trafo3w", "shift_mv_degree", expected_type="f8", default=0.0)
    converter._get_pp_attr.assert_any_call("trafo3w", "shift_lv_degree", expected_type="f8", default=0.0)
    converter._get_pp_attr.assert_any_call("trafo3w", "tap_side", expected_type="O", default=None)
    converter._get_pp_attr.assert_any_call("trafo3w", "tap_neutral", expected_type="f8", default=np.nan)
    converter._get_pp_attr.assert_any_call("trafo3w", "tap_min", expected_type="i4", default=0)
    converter._get_pp_attr.assert_any_call("trafo3w", "tap_max", expected_type="i4", default=0)
    converter._get_pp_attr.assert_any_call("trafo3w", "tap_pos", expected_type="f8", default=np.nan)
    converter._get_pp_attr.assert_any_call("trafo3w", "in_service", expected_type="bool", default=True)
    converter._get_pp_attr.assert_any_call("trafo3w", "vk0_hv_percent", expected_type="f8", default=np.nan)
    converter._get_pp_attr.assert_any_call("trafo3w", "vkr0_hv_percent", expected_type="f8", default=np.nan)
    converter._get_pp_attr.assert_any_call("trafo3w", "vk0_mv_percent", expected_type="f8", default=np.nan)
    converter._get_pp_attr.assert_any_call("trafo3w", "vkr0_mv_percent", expected_type="f8", default=np.nan)
    converter._get_pp_attr.assert_any_call("trafo3w", "vk0_lv_percent", expected_type="f8", default=np.nan)
    converter._get_pp_attr.assert_any_call("trafo3w", "vkr0_lv_percent", expected_type="f8", default=np.nan)
    assert len(converter._get_pp_attr.call_args_list) == 31

    # assignment:
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("trafo3w", two_pp_objs.index))
    pgm.assert_any_call("node_1", _get_pgm_ids("bus", _get_pp_attr("trafo3w", "hv_bus", expected_type="u4")))
    pgm.assert_any_call("node_2", _get_pgm_ids("bus", _get_pp_attr("trafo3w", "mv_bus", expected_type="u4")))
    pgm.assert_any_call("node_3", _get_pgm_ids("bus", _get_pp_attr("trafo3w", "lv_bus", expected_type="u4")))
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
    new=MagicMock(
        return_value=pd.DataFrame(
            {
                "winding_1": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    WindingType.wye_n,
                    WindingType.wye_n,
                    WindingType.wye_n,
                    WindingType.wye_n,
                    0,
                    0,
                ],
                "winding_2": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    WindingType.delta,
                    WindingType.wye_n,
                    WindingType.wye_n,
                    WindingType.delta,
                    0,
                    0,
                ],
                "winding_3": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    WindingType.delta,
                    WindingType.wye_n,
                    WindingType.delta,
                    WindingType.wye_n,
                    0,
                    0,
                ],
            }
        )
    ),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._generate_ids",
    new=MagicMock(return_value=np.arange(1)),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._get_pgm_ids",
    new=MagicMock(return_value=pd.Series([0])),
)
def test_create_pgm_input_transformers3w__default() -> None:
    # Arrange
    pp_net: pp.pandapowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=0.0)
    args = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pp.create_transformer3w_from_parameters(
        pp_net, *args, tap_neutral=12.0, tap_pos=34.0, tap_step_percent=1, tap_side="hv"
    )
    pp.create_transformer3w_from_parameters(
        pp_net, *args, tap_neutral=12.0, tap_pos=34.0, tap_step_percent=1, tap_side="mv"
    )
    pp.create_transformer3w_from_parameters(
        pp_net, *args, tap_neutral=12.0, tap_pos=34.0, tap_step_percent=1, tap_side="lv"
    )
    pp.create_transformer3w_from_parameters(
        pp_net, *args, tap_neutral=12.0, tap_pos=34.0, tap_step_percent=1, tap_side=None
    )
    pp.create_transformer3w_from_parameters(
        pp_net, *args, tap_neutral=np.nan, tap_pos=34.0, tap_step_percent=1, tap_side="hv"
    )
    nan_trafo = pp.create_transformer3w_from_parameters(
        pp_net, *args, tap_neutral=12.0, tap_step_percent=1, tap_pos=np.nan, tap_side="hv"
    )
    pp_net["trafo3w"].loc[nan_trafo, "tap_pos"] = np.nan
    pp.create_transformer3w_from_parameters(
        pp_net, *args, tap_neutral=12.0, tap_pos=34.0, tap_step_percent=np.nan, tap_side="hv"
    )
    pp.create_transformer3w_from_parameters(
        pp_net,
        *args,
        vector_group=None,
        shift_mv_degree=30,
        shift_lv_degree=30,
    )
    pp.create_transformer3w_from_parameters(
        pp_net,
        *args,
        vector_group=None,
        shift_mv_degree=60,
        shift_lv_degree=60,
    )
    pp.create_transformer3w_from_parameters(
        pp_net,
        *args,
        vector_group=None,
        shift_mv_degree=60,
        shift_lv_degree=30,
    )
    pp.create_transformer3w_from_parameters(
        pp_net,
        *args,
        vector_group=None,
        shift_mv_degree=30,
        shift_lv_degree=60,
    )
    pp.create_transformer3w_from_parameters(
        pp_net,
        *args,
        vector_group=None,
        shift_mv_degree=58,
        shift_lv_degree=62,
    )
    pp.create_transformer3w_from_parameters(
        pp_net,
        *args,
        vector_group=None,
        shift_mv_degree=29,
        shift_lv_degree=31,
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
    assert result[4]["tap_side"] == Branch3Side.side_1.value
    assert result[5]["tap_side"] == Branch3Side.side_1.value
    assert result[0]["tap_pos"] == 34.0 != result[0]["tap_nom"]
    assert result[1]["tap_pos"] == 34.0 != result[1]["tap_nom"]
    assert result[2]["tap_pos"] == 34.0 != result[2]["tap_nom"]
    assert result[3]["tap_pos"] == 0 == result[3]["tap_nom"]
    assert result[4]["tap_pos"] == 0 == result[4]["tap_nom"]
    assert result[5]["tap_pos"] == 0 == result[5]["tap_nom"]
    assert result[6]["tap_size"] == 0

    # Default yndd for odd clocks
    assert result[7]["winding_1"] == WindingType.wye_n
    assert result[7]["winding_2"] == WindingType.delta
    assert result[7]["winding_3"] == WindingType.delta
    # Default ynynyn for even clocks
    assert result[8]["winding_1"] == WindingType.wye_n
    assert result[8]["winding_2"] == WindingType.wye_n
    assert result[8]["winding_3"] == WindingType.wye_n
    # Default ynynd for clock_12 even clock_13 odd
    assert result[9]["winding_1"] == WindingType.wye_n
    assert result[9]["winding_2"] == WindingType.wye_n
    assert result[9]["winding_3"] == WindingType.delta
    # Default yndyn for clock_12 odd clock_13 even
    assert result[10]["winding_1"] == WindingType.wye_n
    assert result[10]["winding_2"] == WindingType.delta
    assert result[10]["winding_3"] == WindingType.wye_n

    assert result[11]["clock_12"] == result[11]["clock_13"] == 2
    assert result[12]["clock_12"] == result[12]["clock_13"] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"vk0_hv_percent": 1},
        {"vkr0_hv_percent": 2},
        {"vk0_mv_percent": 3},
        {"vkr0_mv_percent": 4},
        {"vk0_lv_percent": 5},
        {"vkr0_lv_percent": 6},
    ],
)
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
def test_create_pgm_input_transformers3w__zero_sequence(kwargs) -> None:
    # Arrange
    pp_net: pp.pandapowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=0.0)
    args = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pp.create_transformer3w_from_parameters(pp_net, *args, **kwargs)

    converter = PandaPowerConverter()
    converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}

    # Act
    with patch("power_grid_model_io.converters.pandapower_converter.logger") as mock_logger:
        converter._create_pgm_input_three_winding_transformers()
        mock_logger.warning.assert_called_once()


def test_create_pgm_input_three_winding_transformers__tap_at_star_point() -> None:
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


def test_create_pgm_input_three_winding_transformers__tap_dependent_impedance() -> None:
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
    converter._generate_ids.assert_called_once_with("switch", ANY, name="b2b_switches")
    pd.testing.assert_index_equal(converter._generate_ids.call_args_list[0].args[1], pd.Index([1, 3]))

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="link", shape=2)

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
    with pytest.raises(NotImplementedError, match=r"Storage.*not implemented"):
        converter._create_pgm_input_storages()

    # initialization
    mock_init_array.assert_not_called()


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_impedance(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["impedance"] = two_pp_objs

    # Act / Assert
    with pytest.raises(NotImplementedError, match=r"Impedance.*not implemented"):
        converter._create_pgm_input_impedances()

    # initialization
    mock_init_array.assert_not_called()


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
    converter._get_pp_attr.assert_any_call("ward", "bus", expected_type="u4")
    converter._get_pp_attr.assert_any_call("ward", "ps_mw", expected_type="f8")
    converter._get_pp_attr.assert_any_call("ward", "qs_mvar", expected_type="f8")
    converter._get_pp_attr.assert_any_call("ward", "pz_mw", expected_type="f8")
    converter._get_pp_attr.assert_any_call("ward", "qz_mvar", expected_type="f8")
    converter._get_pp_attr.assert_any_call("ward", "in_service", expected_type="bool", default=True)
    assert len(converter._get_pp_attr.call_args_list) == 6

    # assignment:
    for attr in pgm_attr:
        for s in slices:
            pgm[attr].__setitem__.assert_any_call(s, ANY)
        assert len(pgm[attr].__setitem__.call_args_list) == len(slices)

    # result
    assert converter.pgm_input_data["sym_load"] == pgm


def test_create_pgm_input_wards__existing_loads() -> None:
    converter = PandaPowerConverter()
    # Arrange

    pp_net: pp.pandapowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=0.0)
    pp.create_load(pp_net, 0, 0)
    pp.create_ward(pp_net, 0, 0, 0, 0, 0)

    converter.pp_input_data = pp_net

    # Act
    converter._create_pgm_input_nodes()
    converter._create_pgm_input_sym_loads()
    converter._create_pgm_input_wards()

    # assert
    assert len(converter.pgm_input_data["sym_load"]) == 5


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_xward(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["xward"] = two_pp_objs

    # Act / Assert
    with pytest.raises(NotImplementedError, match=r"Extended Ward.*not implemented"):
        converter._create_pgm_input_xwards()

    # initialization
    mock_init_array.assert_not_called()


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_generators(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["gen"] = two_pp_objs

    # Act / Assert
    with pytest.raises(NotImplementedError, match=r"Generators.*not implemented"):
        converter._create_pgm_input_generators()

    # initialization
    mock_init_array.assert_not_called()


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_dclines(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["dcline"] = two_pp_objs

    # Act / Assert
    with pytest.raises(NotImplementedError, match=r"DC line .*not implemented"):
        converter._create_pgm_input_dclines()

    # initialization
    mock_init_array.assert_not_called()


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
@patch("power_grid_model_io.converters.pandapower_converter.np.divide", new=lambda x, _, **kwargs: x)
@patch("power_grid_model_io.converters.pandapower_converter.np.power", new=lambda x, _, **kwargs: x)
@patch("power_grid_model_io.converters.pandapower_converter.np.sqrt", new=lambda x, **kwargs: x)
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
    converter._get_pp_attr.assert_any_call("motor", "bus", expected_type="i8")
    converter._get_pp_attr.assert_any_call("motor", "pn_mech_mw", expected_type="f8")
    converter._get_pp_attr.assert_any_call("motor", "cos_phi", expected_type="f8")
    converter._get_pp_attr.assert_any_call("motor", "efficiency_percent", expected_type="f8")
    converter._get_pp_attr.assert_any_call("motor", "loading_percent", expected_type="f8")
    converter._get_pp_attr.assert_any_call("motor", "scaling", expected_type="f8")
    converter._get_pp_attr.assert_any_call("motor", "in_service", expected_type="bool", default=True)
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


def test_create_pgm_input_motors__existing_loads() -> None:
    converter = PandaPowerConverter()
    # Arrange

    pp_net: pp.pandapowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=0.0)
    pp.create_load(pp_net, 0, 0)
    pp.create_motor(pp_net, 0, 0, 0)

    converter.pp_input_data = pp_net

    # Act
    converter._create_pgm_input_nodes()
    converter._create_pgm_input_sym_loads()
    converter._create_pgm_input_motors()

    # assert
    assert len(converter.pgm_input_data["sym_load"]) == 4


@pytest.mark.parametrize(
    "create_fn",
    [
        PandaPowerConverter._create_pgm_input_sources,
        PandaPowerConverter._create_pgm_input_shunts,
        PandaPowerConverter._create_pgm_input_lines,
        PandaPowerConverter._create_pgm_input_sym_gens,
        PandaPowerConverter._create_pgm_input_sym_loads,
        PandaPowerConverter._create_pgm_input_asym_gens,
        PandaPowerConverter._create_pgm_input_asym_loads,
        PandaPowerConverter._create_pgm_input_impedances,
        PandaPowerConverter._create_pgm_input_links,
        PandaPowerConverter._create_pgm_input_motors,
        PandaPowerConverter._create_pgm_input_nodes,
        PandaPowerConverter._create_pgm_input_storages,
        PandaPowerConverter._create_pgm_input_three_winding_transformers,
        PandaPowerConverter._create_pgm_input_transformers,
        PandaPowerConverter._create_pgm_input_wards,
        PandaPowerConverter._create_pgm_input_xwards,
        PandaPowerConverter._create_pgm_input_sources,
        PandaPowerConverter._create_pgm_input_generators,
        PandaPowerConverter._create_pgm_input_dclines,
    ],
)
def test_create_pp_input_object__empty(create_fn: Callable[[PandaPowerConverter], None]):
    # Arrange: No table
    converter = PandaPowerConverter()
    converter.pp_input_data = pp.create_empty_network()

    # Act / Assert
    with patch("power_grid_model_io.converters.pandapower_converter.initialize_array") as mock_init_array:
        create_fn(converter)
        mock_init_array.assert_not_called()


def test_generate_ids():
    converter = PandaPowerConverter()
    test_table = pd.DataFrame(
        [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
        columns=["col1", "col2", "col3"],
        index=[11, 12, 13],
    )
    converter.pp_input_data["test_table"] = test_table
    converter.next_idx = 1
    pgm_idx_actual = converter._generate_ids("test_table", test_table.index, name="ids_name")
    pgm_idx_expected = np.array([1, 2, 3], dtype=np.int32)

    assert converter.next_idx == 4
    pd.testing.assert_series_equal(
        converter.idx[("test_table", "ids_name")], pd.Series(pgm_idx_expected, index=test_table.index)
    )
    pd.testing.assert_series_equal(
        converter.idx_lookup[("test_table", "ids_name")], pd.Series(test_table.index, index=pgm_idx_expected)
    )
    np.testing.assert_array_equal(pgm_idx_actual, pgm_idx_expected)


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


def test_get_trafo_winding_types__vector_group_missing():
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {"trafo": pd.DataFrame([1, 2, 3], columns=["id"])}
    expected = pd.DataFrame(np.full((3, 2), np.nan), columns=["winding_from", "winding_to"])

    # Act
    actual = converter.get_trafo_winding_types()

    # Assert
    pd.testing.assert_frame_equal(actual, expected)


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


def test_get_trafo3w_winding_types__vector_group_missing():
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {"trafo3w": pd.DataFrame([1, 2, 3], columns=["id"])}
    expected = pd.DataFrame(np.full((3, 3), np.nan), columns=["winding_1", "winding_2", "winding_3"])

    # Act
    actual = converter.get_trafo3w_winding_types()

    # Assert
    pd.testing.assert_frame_equal(actual, expected)


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


def test_get_pp_attr_attribute_exists():
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {
        "trafo3w": pd.DataFrame([[2, 32, 31, 315]], columns=["index", "hv_bus", "mv_bus", "lv_bus"])
    }
    expected = np.array(32)

    # Act
    actual = converter._get_pp_attr("trafo3w", "hv_bus", expected_type="u4")

    # Assert
    np.testing.assert_array_equal(actual, expected)


def test_get_pp_attr_attribute_doesnt_exist():
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {"trafo3w": pd.DataFrame([[2, 31, 315]], columns=["index", "mv_bus", "lv_bus"])}

    # Act / Assert
    with pytest.raises(KeyError):
        converter._get_pp_attr("trafo3w", "hv_bus", expected_type="u4")


def test_get_pp_attr_use_default():
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {"trafo3w": pd.DataFrame([[2, 31, 315]], columns=["index", "mv_bus", "lv_bus"])}
    expected = np.array(625)

    # Act
    actual = converter._get_pp_attr("trafo3w", "hv_bus", expected_type="u4", default=625)

    # Assert
    np.testing.assert_array_equal(actual, expected)
