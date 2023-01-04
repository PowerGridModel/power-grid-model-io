# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import Callable, Tuple
from unittest.mock import ANY, MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest
from power_grid_model import WindingType, initialize_array
from power_grid_model.data_types import SingleDataset
from power_grid_model.utils import import_json_data

from power_grid_model_io.converters.pandapower_converter import PandaPowerConverter, PandaPowerData
from power_grid_model_io.data_types import ExtraInfoLookup

from ...data.pandapower.pp_validation import pp_net
from ...utils import MockDf, MockFn, assert_struct_array_equal

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pandapower"


@pytest.fixture
def pp_example_simple() -> Tuple[PandaPowerData, float]:
    net = pp_net()
    return net, net.f_hz


@pytest.fixture
def pgm_example_simple() -> SingleDataset:
    return import_json_data(DATA_DIR / "pgm_input_data.json", data_type="input", ignore_extra=True)


def _generate_ids(*args):
    return MockFn("_generate_ids", *args)


def _get_ids(*args):
    return MockFn("_get_ids", *args)


def _get_pp_attr(*args):
    return MockFn("_get_pp_attr", *args)


def get_switch_states(*args):
    return MockFn("get_switch_states", *args)


@pytest.fixture
def converter() -> PandaPowerConverter:
    converter = PandaPowerConverter()
    converter._generate_ids = MagicMock(side_effect=_generate_ids)  # type: ignore
    converter._get_ids = MagicMock(side_effect=_get_ids)  # type: ignore
    converter._get_pp_attr = MagicMock(side_effect=_get_pp_attr)  # type: ignore
    converter.get_switch_states = MagicMock(side_effect=get_switch_states)  # type: ignore
    return converter


@pytest.fixture
def two_pp_objs() -> MockDf:
    return MockDf(2)


@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._create_input_data")
def test_parse_data(create_input_data_mock: MagicMock):
    # Arrange
    converter = PandaPowerConverter()

    def create_input_data():
        converter.pgm_data = {"node": np.array([])}

    create_input_data_mock.side_effect = create_input_data

    # Act
    result = converter._parse_data(data={"bus": pd.DataFrame()}, data_type="input", extra_info=None)

    # Assert
    create_input_data_mock.assert_called_once_with()
    assert len(converter.pp_data) == 1 and "bus" in converter.pp_data
    assert len(converter.pgm_data) == 1 and "node" in converter.pgm_data
    assert len(result) == 1 and "node" in result


@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._create_input_data")
def test_parse_data__extra_info(create_input_data_mock: MagicMock):
    # Arrange
    converter = PandaPowerConverter()

    def create_input_data():
        converter.idx_lookup[("bus", None)] = pd.Series([101, 102, 103], index=[0, 1, 2])
        converter.idx_lookup[("load", "const_current")] = pd.Series([201, 202, 203], index=[3, 4, 5])

    create_input_data_mock.side_effect = create_input_data

    # Act
    extra_info: ExtraInfoLookup = {}
    converter._parse_data(data={}, data_type="input", extra_info=extra_info)

    # Assert
    assert len(extra_info) == 6
    assert extra_info[0] == {"id_reference": {"table": "bus", "index": 101}}
    assert extra_info[1] == {"id_reference": {"table": "bus", "index": 102}}
    assert extra_info[2] == {"id_reference": {"table": "bus", "index": 103}}
    assert extra_info[3] == {"id_reference": {"table": "load", "name": "const_current", "index": 201}}
    assert extra_info[4] == {"id_reference": {"table": "load", "name": "const_current", "index": 202}}
    assert extra_info[5] == {"id_reference": {"table": "load", "name": "const_current", "index": 203}}


def test_parse_data__update_data():
    # Arrange
    converter = PandaPowerConverter()

    # Act/Assert
    with pytest.raises(ValueError):
        converter._parse_data(data={}, data_type="update", extra_info=None)


def test_create_input_data():
    # Arrange
    converter = MagicMock()

    # Act
    PandaPowerConverter._create_input_data(self=converter)  # type: ignore

    # Assert
    assert len(converter.method_calls) == 14
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
    converter._create_pgm_input_ward.assert_called_once_with()
    converter._create_pgm_input_xward.assert_called_once_with()
    converter._create_pgm_input_motor.assert_called_once_with()


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
    converter.pp_data[table] = pd.DataFrame()  # type: ignore

    # Act
    create_fn(converter)

    # Assert
    mock_init_array.assert_not_called()


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_nodes(mock_init_array: MagicMock, two_pp_objs: MockDf, converter):
    # Arrange
    converter.pp_data["bus"] = two_pp_objs

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
    assert converter.pgm_data["node"] == mock_init_array.return_value


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_lines(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_data["line"] = two_pp_objs

    # Act
    converter._create_pgm_input_lines()

    # Assert

    # administration
    converter.get_switch_states.assert_called_once_with("line")
    converter._generate_ids.assert_called_once_with("line", two_pp_objs.index)
    converter._get_ids.assert_any_call("bus", two_pp_objs["from_bus"])
    converter._get_ids.assert_any_call("bus", two_pp_objs["to_bus"])

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="line", shape=2)

    # retrieval
    converter._get_pp_attr.assert_any_call("line", "in_service")
    converter._get_pp_attr.assert_any_call("line", "length_km")
    converter._get_pp_attr.assert_any_call("line", "parallel")
    converter._get_pp_attr.assert_any_call("line", "r_ohm_per_km")
    converter._get_pp_attr.assert_any_call("line", "x_ohm_per_km")
    converter._get_pp_attr.assert_any_call("line", "c_nf_per_km")
    converter._get_pp_attr.assert_any_call("line", "g_us_per_km")
    converter._get_pp_attr.assert_any_call("line", "max_i_ka")
    converter._get_pp_attr.assert_any_call("line", "df")
    assert len(converter._get_pp_attr.call_args_list) == 9

    # assignment
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("line", two_pp_objs.index))
    pgm.assert_any_call("from_node", _get_ids("bus", two_pp_objs["from_bus"]))
    pgm.assert_any_call("from_status", _get_pp_attr("line", "in_service") & get_switch_states("line")["from"])
    pgm.assert_any_call("to_node", _get_ids("bus", two_pp_objs["to_bus"]))
    pgm.assert_any_call("to_status", _get_pp_attr("line", "in_service") & get_switch_states("line")["to"])
    pgm.assert_any_call(
        "r1",
        _get_pp_attr("line", "r_ohm_per_km") * (_get_pp_attr("line", "length_km") / _get_pp_attr("line", "parallel")),
    )
    pgm.assert_any_call(
        "x1",
        _get_pp_attr("line", "x_ohm_per_km") * (_get_pp_attr("line", "length_km") / _get_pp_attr("line", "parallel")),
    )
    pgm.assert_any_call(
        "c1",
        _get_pp_attr("line", "c_nf_per_km")
        * _get_pp_attr("line", "length_km")
        * _get_pp_attr("line", "parallel")
        * 1e-9,
    )
    pgm.assert_any_call(
        "tan1", _get_pp_attr("line", "g_us_per_km") / _get_pp_attr("line", "c_nf_per_km") / (np.pi / 10)
    )
    pgm.assert_any_call(
        "i_n", _get_pp_attr("line", "max_i_ka") * 1e3 * _get_pp_attr("line", "df") * _get_pp_attr("line", "parallel")
    )

    # result
    assert converter.pgm_data["line"] == mock_init_array.return_value


def test_create_pgm_input_sources(pp_example_simple: Tuple[PandaPowerData, float], pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter(system_frequency=pp_example_simple[1])
    converter.pp_data = pp_example_simple[0]
    converter.idx = {("bus", None): pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 7

    # Act
    converter._create_pgm_input_sources()

    # Assert
    assert_struct_array_equal(converter.pgm_data["source"], pgm_example_simple["source"])


def test_create_pgm_input_sym_loads(pp_example_simple: Tuple[PandaPowerData, float], pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter(system_frequency=pp_example_simple[1])
    converter.pp_data = pp_example_simple[0]
    converter.idx = {("bus", None): pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 8

    # Act
    converter._create_pgm_input_sym_loads()

    # Assert
    assert_struct_array_equal(converter.pgm_data["sym_load"], pgm_example_simple["sym_load"])


def test_create_pgm_input_transformers(
    pp_example_simple: Tuple[PandaPowerData, float], pgm_example_simple: SingleDataset
):
    # Arrange
    converter = PandaPowerConverter(system_frequency=pp_example_simple[1])
    converter.pp_data = pp_example_simple[0]
    converter.idx = {("bus", None): pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 12

    # Act
    converter._create_pgm_input_transformers()

    # Assert
    assert_struct_array_equal(converter.pgm_data["transformer"], pgm_example_simple["transformer"])


def test_create_pgm_input_shunts(pp_example_simple: Tuple[PandaPowerData, float], pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter(system_frequency=pp_example_simple[1])
    converter.pp_data = pp_example_simple[0]
    converter.idx = {("bus", None): pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 11

    # Act
    converter._create_pgm_input_shunts()

    # Assert
    assert_struct_array_equal(converter.pgm_data["shunt"], pgm_example_simple["shunt"])


def test_create_pgm_input_sym_gens(pp_example_simple: Tuple[PandaPowerData, float], pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter(system_frequency=pp_example_simple[1])
    converter.pp_data = pp_example_simple[0]
    converter.idx = {("bus", None): pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 13

    # Act
    converter._create_pgm_input_sym_gens()

    # Assert
    assert_struct_array_equal(converter.pgm_data["sym_gen"], pgm_example_simple["sym_gen"])


def test_create_pgm_input_three_winding_transformers(
    pp_example_simple: Tuple[PandaPowerData, float], pgm_example_simple: SingleDataset
):
    # Arrange
    converter = PandaPowerConverter(system_frequency=pp_example_simple[1])
    converter.pp_data = pp_example_simple[0]
    converter.idx = {("bus", None): pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 14

    # Act
    converter._create_pgm_input_three_winding_transformers()

    # Assert
    assert_struct_array_equal(
        converter.pgm_data["three_winding_transformer"], pgm_example_simple["three_winding_transformer"]
    )


def test_create_pgm_input_links(pp_example_simple: Tuple[PandaPowerData, float], pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter(system_frequency=pp_example_simple[1])
    converter.pp_data = pp_example_simple[0]
    converter.idx = {("bus", None): pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 15

    # Act
    converter._create_pgm_input_links()

    # Assert
    assert_struct_array_equal(converter.pgm_data["link"], pgm_example_simple["link"])


# def test__pp_buses_output(pgm_output: SingleDataset, pp_expected_output: SingleDataset):
#     # Arrange
#     converter = PandaPowerConverter(system_frequency=50)
#     converter.pgm_output_data = pgm_output
#     converter.idx_lookup["bus"] = pd.Series([321, 5, 4, 3, 1, 6])
#
#     # Act
#     converter._pp_buses_output()
#
#     # Assert
#     np.testing.assert_array_equal(converter.pp_output_data["node"], pp_expected_output["bus"])


# def test__pp_shunts_output(pgm_output: SingleDataset, pp_expected_output: SingleDataset):
#     # Arrange
#     converter = PandaPowerConverter(system_frequency=50)
#     converter.pgm_output_data = pgm_output
#
#     source = initialize_array("input", "source", 1)
#     source["id"] = [10]
#     source["node"] = [1]
#     source["status"] = [1]
#     source["u_ref"] = [1.0]
#
#     converter.pgm_data["source"] = source
#     converter.idx_lookup["bus"] = pd.Series([321, 5, 4, 3, 1, 6])
#
#     converter.pgm_nodes_lookup = pd.DataFrame()
#
#     # Act
#     converter._pp_buses_output()
#
#     # Assert
#     np.testing.assert_array_equal(converter.pp_output_data["node"], pp_expected_output["bus"])


def test_get_index__key_error():
    # Arrange
    converter = PandaPowerConverter()

    # Act / Assert
    with pytest.raises(KeyError, match=r"index.*bus"):
        converter._get_ids(pp_table="bus", pp_idx=pd.Series(dtype=np.int32))


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
    pp_trafo_tap_side = pd.Series(["hv", "lv", "lv", "hv"])
    expected_tap_side = np.array([0, 1, 1, 0], dtype=np.int8)

    # Act
    actual_tap_side = PandaPowerConverter._get_transformer_tap_side(pp_trafo_tap_side)

    # Assert
    np.testing.assert_array_equal(actual_tap_side, expected_tap_side)


def test_get_3wtransformer_tap_side():
    # Arrange
    pp_trafo3w_tap_side = pd.Series(["hv", "mv", "lv", "mv", "lv", "hv", "lv"])
    expected_tap_side = np.array([0, 1, 2, 1, 2, 0, 2], dtype=np.int8)

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
    converter.pp_data = {"trafo": pd.DataFrame([(1, "Dyn"), (2, "YNd"), (3, "Dyn")], columns=["id", "vector_group"])}
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
    converter.pp_data = {
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
    converter.pp_data = {
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
    converter.pp_data = {
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
    converter.pp_data = {"trafo": pd.DataFrame([(1, "ADyn")], columns=["id", "vector_group"])}

    # Act / Assert
    with pytest.raises(ValueError):
        converter.get_trafo_winding_types()


def test_get_trafo3w_winding_types__value_error():
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_data = {"trafo3w": pd.DataFrame([(1, "ADyndrr")], columns=["id", "vector_group"])}

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

    expected_id = 345

    # Act
    actual_id = converter.get_id("line", 1)

    # Assert
    np.testing.assert_array_equal(actual_id, expected_id)


@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter.get_individual_switch_states")
def test_get_switch_states_lines(mock_get_individual_switch_states: MagicMock):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_data = {
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
    converter.pp_data = {
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
    converter.pp_data = {
        "trafo3w": pd.DataFrame([[2, 32, 31, 315]], columns=["index", "hv_bus", "mv_bus", "lv_bus"]),
        "switch": pd.DataFrame(
            [[101, 315, "t3", 2, False], [321, 32, "t3", 2, False]],
            columns=["index", "bus", "et", "element", "closed"],
        ),
    }
    mock_get_individual_switch_states.side_effect = [False, True, False]

    expected = pd.DataFrame(data=([False], [True], [False]))

    # Act
    actual = converter.get_trafo3w_switch_states(converter.pp_data["trafo3w"])

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

    converter.pgm_data = {
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
    converter.pgm_data["line"]["from_node"] = [0, 1, 1]
    converter.pgm_data["line"]["to_node"] = [1, 2, 3]
    converter.pgm_data["link"]["from_node"] = [0, 1]
    converter.pgm_data["link"]["to_node"] = [1, 2]
    converter.pgm_data["transformer"]["from_node"] = [0, 1]
    converter.pgm_data["transformer"]["to_node"] = [1, 2]
    converter.pgm_data["three_winding_transformer"]["node_1"] = [0, 1]
    converter.pgm_data["three_winding_transformer"]["node_2"] = [1, 2]
    converter.pgm_data["three_winding_transformer"]["node_3"] = [2, 3]
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
    assert pp_buses["p_mw"][101] * 1e6 == 1.0 + 10.0 + 100.0 + 1000.0
    assert pp_buses["p_mw"][102] * 1e6 == 2.0 + 4.0 - 1.0 + 20.0 - 10.0 + 200.0 - 100.0 + 10000.0 + 2000.0
    assert pp_buses["p_mw"][103] * 1e6 == -2.0 - 20.0 - 200.0 + 20000.0 + 4000.0
    assert pp_buses["p_mw"][104] * 1e6 == -4.0 + 40000.0
    assert pp_buses["q_mvar"][101] * 1e6 == 0.1 + 0.01 + 0.001 + 0.0001
    assert pp_buses["q_mvar"][102] * 1e6 == 0.2 + 0.4 - 0.1 + 0.02 - 0.01 + 0.002 - 0.001 + 0.00001 + 0.0002
    assert pp_buses["q_mvar"][103] * 1e6 == -0.2 - 0.02 - 0.002 + 0.00002 + 0.0004
    assert pp_buses["q_mvar"][104] * 1e6 == -0.4 + 0.00004


def test_get_pp_attr_attribute_exists():
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_data = {"trafo3w": pd.DataFrame([[2, 32, 31, 315]], columns=["index", "hv_bus", "mv_bus", "lv_bus"])}
    expected = np.array(32)

    # Act
    actual = converter._get_pp_attr("trafo3w", "hv_bus")

    # Assert
    np.testing.assert_array_equal(actual, expected)


def test_get_pp_attr_attribute_doesnt_exist():
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_data = {"trafo3w": pd.DataFrame([[2, 31, 315]], columns=["index", "mv_bus", "lv_bus"])}

    # Act / Assert
    with pytest.raises(KeyError):
        converter._get_pp_attr("trafo3w", "hv_bus")


def test_get_pp_attr_use_default():
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_data = {"trafo3w": pd.DataFrame([[2, 31, 315]], columns=["index", "mv_bus", "lv_bus"])}
    expected = np.array(625)

    # Act
    actual = converter._get_pp_attr("trafo3w", "hv_bus", 625)

    # Assert
    np.testing.assert_array_equal(actual, expected)


def test_get_pp_attr_from_std():
    # Arrange
    converter = PandaPowerConverter()
    converter._std_types = {"trafo3w": {"std_trafo3w_1": {"hv_bus": 964}}}
    converter.pp_data = {
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
    converter.pp_data = {
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
    converter.pp_data = {
        "trafo3w": pd.DataFrame([[2, 31, 315, "std_trafo3w_1"]], columns=["index", "mv_bus", "lv_bus", "std_type"])
    }

    # Act/Assert
    with pytest.raises(KeyError):
        converter._get_pp_attr("trafo3w", "hv_bus")
