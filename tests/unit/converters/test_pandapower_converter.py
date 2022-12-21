# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import json
from pathlib import Path
from typing import Callable, Tuple
from unittest.mock import ANY, MagicMock, call, patch

import numpy as np
import pandapower as pp
import pandas as pd
import pytest
from power_grid_model import CalculationMethod, PowerGridModel, WindingType, initialize_array
from power_grid_model.data_types import SingleDataset
from power_grid_model.utils import import_input_data, import_json_data

from power_grid_model_io.converters.pandapower_converter import PandaPowerConverter, PandaPowerData
from power_grid_model_io.data_types import ExtraInfoLookup

from ...utils import assert_struct_array_equal

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pandapower"


@pytest.fixture
def pp_example_simple() -> Tuple[PandaPowerData, float]:
    #  (ext #1)         shunt - [104]  - 3w - [105] - sym_gen
    #   |                                |
    #  [101] -/- -OO- [102] ----/----- [103]
    #   |                                |
    #  -/-                          (load #31)
    #   |
    #  [106]
    net = pp.create_empty_network(f_hz=50)
    pp.create_bus(net, index=101, vn_kv=110)
    pp.create_bus(net, index=102, vn_kv=20)
    pp.create_bus(net, index=103, vn_kv=20)
    pp.create_bus(net, index=104, vn_kv=30.1)
    pp.create_bus(net, index=105, vn_kv=60)
    pp.create_bus(net, index=106, vn_kv=34)
    pp.create_ext_grid(
        net, index=1, in_service=True, bus=101, vm_pu=31.02, s_sc_max_mva=3.0, rx_max=0.6, va_degree=61.2
    )
    pp.create_transformer_from_parameters(
        net,
        index=101,
        hv_bus=101,
        lv_bus=102,
        i0_percent=38.0,
        pfe_kw=11.6,
        vkr_percent=0.322,
        sn_mva=40,
        vn_lv_kv=22.0,
        vn_hv_kv=110.0,
        vk_percent=17.8,
        vector_group="Dyn",
        shift_degree=30,
        tap_side="hv",
        tap_pos=2,
        tap_min=1,
        tap_max=3,
        tap_step_percent=30,
        tap_neutral=2,
        parallel=3,
    )
    pp.create_line(
        net, index=101, from_bus=103, to_bus=102, length_km=1.23, parallel=2, df=10, std_type="NAYY 4x150 SE"
    )
    pp.create_load(
        net, index=101, bus=103, p_mw=2.5, q_mvar=0.24, const_i_percent=26.0, const_z_percent=51.0, cos_phi=2
    )
    pp.create_switch(net, index=101, et="l", bus=103, element=101, closed=False)
    pp.create_switch(net, index=3021, et="b", bus=101, element=106, closed=True)
    pp.create_switch(net, index=321, et="t", bus=101, element=101, closed=False)
    pp.create_shunt(net, index=1201, in_service=True, bus=104, p_mw=2.1, q_mvar=31.5, step=3)
    pp.create_sgen(net, index=31, bus=105, p_mw=6.21, q_mvar=20.1)
    pp.create_transformer3w_from_parameters(
        net,
        index=102,
        hv_bus=103,
        mv_bus=105,
        lv_bus=104,
        in_service=True,
        vn_hv_kv=110.0,
        vn_mv_kv=50.0,
        vn_lv_kv=22.0,
        sn_hv_mva=40,
        sn_mv_mva=100,
        sn_lv_mva=50,
        vk_hv_percent=20,
        vk_mv_percent=60,
        vk_lv_percent=35,
        vkr_hv_percent=10,
        vkr_mv_percent=20,
        vkr_lv_percent=40,
        i0_percent=38,
        pfe_kw=11.6,
        vector_group="Dynz",
        shift_mv_degree=30,
        shift_lv_degree=60,
        tap_pos=2,
        tap_side="lv",
        tap_min=1,
        tap_max=3,
        tap_step_percent=30,
        tap_neutral=2,
    )

    components = {component: net[component] for component in net if isinstance(net[component], (pd.DataFrame, float))}
    return components, net.f_hz


@pytest.fixture
def pgm_example_simple() -> SingleDataset:
    return import_input_data(DATA_DIR / "input_data.json")


@pytest.fixture
def pgm_output() -> SingleDataset:
    f = open(DATA_DIR / "pgm_output_data.json")
    return json.load(f)


@pytest.fixture
def pp_expected_output() -> SingleDataset:
    f = open(DATA_DIR / "pp_output_data.json")
    return json.load(f)


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
    assert len(converter.method_calls) == 9
    converter._create_pgm_input_nodes.assert_called_once_with()
    converter._create_pgm_input_lines.assert_called_once_with()
    converter._create_pgm_input_sources.assert_called_once_with()
    converter._create_pgm_input_sym_loads.assert_called_once_with()
    converter._create_pgm_input_shunts.assert_called_once_with()
    converter._create_pgm_input_transformers.assert_called_once_with()
    converter._create_pgm_input_sym_gens.assert_called_once_with()
    converter._create_pgm_input_three_winding_transformers.assert_called_once_with()
    converter._create_pgm_input_links.assert_called_once_with()


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


def test_create_pgm_input_nodes(pp_example_simple: Tuple[PandaPowerData, float], pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter(system_frequency=pp_example_simple[1])
    converter.pp_data = pp_example_simple[0]

    # Act
    converter._create_pgm_input_nodes()

    # Assert
    np.testing.assert_array_equal(converter.pgm_data["node"], pgm_example_simple["node"])


def test_create_pgm_input_lines(pp_example_simple: Tuple[PandaPowerData, float], pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter(system_frequency=pp_example_simple[1])
    converter.pp_data = pp_example_simple[0]
    converter.idx = {("bus", None): pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 5

    # Act
    converter._create_pgm_input_lines()

    # Assert
    assert_struct_array_equal(converter.pgm_data["line"], pgm_example_simple["line"])


def test_create_pgm_input_sources(pp_example_simple: Tuple[PandaPowerData, float], pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter(system_frequency=pp_example_simple[1])
    converter.pp_data = pp_example_simple[0]
    converter.idx = {("bus", None): pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 6

    # Act
    converter._create_pgm_input_sources()

    # Assert
    assert_struct_array_equal(converter.pgm_data["source"], pgm_example_simple["source"])


def test_create_pgm_input_sym_loads(pp_example_simple: Tuple[PandaPowerData, float], pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter(system_frequency=pp_example_simple[1])
    converter.pp_data = pp_example_simple[0]
    converter.idx = {("bus", None): pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 7

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
    converter.next_idx = 10

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
    converter.next_idx = 12

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
    converter.next_idx = 13

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
    converter.next_idx = 14

    # Act
    converter._create_pgm_input_links()

    # Assert
    assert_struct_array_equal(converter.pgm_data["link"], pgm_example_simple["link"])


def test__pp_buses_output(pgm_output: SingleDataset, pp_expected_output: SingleDataset):
    # Arrange
    converter = PandaPowerConverter(system_frequency=50)
    converter.pgm_output_data = pgm_output
    converter.idx_lookup["bus"] = pd.Series([321, 5, 4, 3, 1, 6])

    # Act
    converter._pp_buses_output()

    # Assert
    np.testing.assert_array_equal(converter.pp_output_data["node"], pp_expected_output["bus"])


def test__pp_shunts_output(pgm_output: SingleDataset, pp_expected_output: SingleDataset):
    # Arrange
    converter = PandaPowerConverter(system_frequency=50)
    converter.pgm_output_data = pgm_output

    source = initialize_array("input", "source", 1)
    source["id"] = [10]
    source["node"] = [1]
    source["status"] = [1]
    source["u_ref"] = [1.0]

    converter.pgm_data["source"] = source
    converter.idx_lookup["bus"] = pd.Series([321, 5, 4, 3, 1, 6])

    converter.pgm_nodes_lookup = pd.DataFrame()

    # Act
    converter._pp_buses_output()

    # Assert
    np.testing.assert_array_equal(converter.pp_output_data["node"], pp_expected_output["bus"])


def test_get_index__key_error():
    # Arrange
    converter = PandaPowerConverter()

    # Act / Assert
    with pytest.raises(KeyError, match=r"index.*bus"):
        converter._get_ids(pp_table="bus", pp_idx=pd.Series())


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
        [[101, 101], [201, 103]],
        columns=["index", "hv_bus"],
    )
    pp_switches = pd.DataFrame(
        [[101, 101, 101, False], [102, 103, 201, True]],
        columns=["index", "bus", "element", "closed"],
    )

    expected_state = np.array([False, True], dtype=np.float64)

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
        "line": pd.DataFrame([[101, 31]], index=[21], columns=["from_bus", "to_bus"]),
        "switch": pd.DataFrame(
            [[101, "l", 21, False], [102, "x", 22, True]],
            index=[1001, 1002],
            columns=["bus", "et", "element", "closed"],
        ),
    }
    mock_get_individual_switch_states.side_effect = [False, True]

    expected = pd.DataFrame(data=([False], [True]))

    # Act
    actual = converter.get_switch_states("line")

    # Assert
    pd.testing.assert_frame_equal(actual, expected)

    assert len(mock_get_individual_switch_states.call_args_list) == 2

    assert mock_get_individual_switch_states.call_args_list[0] == call(ANY, ANY, "from_bus")
    pd.testing.assert_frame_equal(
        mock_get_individual_switch_states.call_args_list[0].args[0],
        pd.DataFrame([[101, 31, 21]], index=[21], columns=["from_bus", "to_bus", "index"]),
    )
    pd.testing.assert_frame_equal(
        mock_get_individual_switch_states.call_args_list[0].args[1],
        pd.DataFrame(
            [[21, 101, False]],
            index=[1001],
            columns=["element", "bus", "closed"],
        ),
    )

    assert mock_get_individual_switch_states.call_args_list[1] == call(ANY, ANY, "to_bus")
    pd.testing.assert_frame_equal(
        mock_get_individual_switch_states.call_args_list[1].args[0],
        pd.DataFrame([[101, 31, 21]], index=[21], columns=["from_bus", "to_bus", "index"]),
    )
    pd.testing.assert_frame_equal(
        mock_get_individual_switch_states.call_args_list[1].args[1],
        pd.DataFrame(
            [[21, 101, False]],
            index=[1001],
            columns=["element", "bus", "closed"],
        ),
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
    mock_get_individual_switch_states.side_effect = [True, False]

    expected = pd.DataFrame(data=([True], [False]))

    # Act
    actual = converter.get_switch_states("trafo")

    # Assert
    pd.testing.assert_frame_equal(actual, expected)
    assert len(mock_get_individual_switch_states.call_args_list) == 2


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
    converter.idx_lookup = {"bus": pd.Series([101, 102, 103, 104], index=[0, 1, 2, 3], dtype=np.int32)}
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
