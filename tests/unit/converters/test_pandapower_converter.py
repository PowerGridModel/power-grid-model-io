# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandapower as pp
import pandas as pd
import pytest
from power_grid_model import initialize_array
from power_grid_model.data_types import SingleDataset
from power_grid_model.utils import import_input_data

from power_grid_model_io.converters.pandapower_converter import PandaPowerConverter, PandasData

from ...utils import assert_struct_array_equal

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pandapower"


@pytest.fixture
def pp_example_simple() -> Tuple[PandasData, Dict[str, float]]:
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
    pp.create_ext_grid(net, index=1, in_service=True, bus=101, vm_pu=31.02, s_sc_max_mva=3.0)
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
    grid_config = {"f_hz": net.f_hz}
    return components, grid_config


@pytest.fixture
def pgm_example_simple() -> SingleDataset:
    return import_input_data(DATA_DIR / "input_data.json")


def test_create_pgm_input_nodes(
    pp_example_simple: Tuple[PandasData, Dict[str, float]], pgm_example_simple: SingleDataset
):
    # Arrange
    converter = PandaPowerConverter(grid_config=pp_example_simple[1])
    converter.pp_data = pp_example_simple[0]

    # Act
    converter._create_pgm_input_nodes()

    # Assert
    np.testing.assert_array_equal(converter.pgm_data["node"], pgm_example_simple["node"])


def test_create_pgm_input_lines(
    pp_example_simple: Tuple[PandasData, Dict[str, float]], pgm_example_simple: SingleDataset
):
    # Arrange
    converter = PandaPowerConverter(grid_config=pp_example_simple[1])
    converter.pp_data = pp_example_simple[0]
    converter.idx = {"bus": pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 5

    # Act
    converter._create_pgm_input_lines()

    # Assert
    assert_struct_array_equal(converter.pgm_data["line"], pgm_example_simple["line"])


def test_create_pgm_input_sources(
    pp_example_simple: Tuple[PandasData, Dict[str, float]], pgm_example_simple: SingleDataset
):
    # Arrange
    converter = PandaPowerConverter(grid_config=pp_example_simple[1])
    converter.pp_data = pp_example_simple[0]
    converter.idx = {"bus": pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 6

    # Act
    converter._create_pgm_input_sources()

    # Assert
    assert_struct_array_equal(converter.pgm_data["source"], pgm_example_simple["source"])


def test_create_pgm_input_sym_loads(
    pp_example_simple: Tuple[PandasData, Dict[str, float]], pgm_example_simple: SingleDataset
):
    # Arrange
    converter = PandaPowerConverter(grid_config=pp_example_simple[1])
    converter.pp_data = pp_example_simple[0]
    converter.idx = {"bus": pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 7

    # Act
    converter._create_pgm_input_sym_loads()

    # Assert
    assert_struct_array_equal(converter.pgm_data["sym_load"], pgm_example_simple["sym_load"])


def test_create_pgm_input_transformers(
    pp_example_simple: Tuple[PandasData, Dict[str, float]], pgm_example_simple: SingleDataset
):
    # Arrange
    converter = PandaPowerConverter(grid_config=pp_example_simple[1])
    converter.pp_data = pp_example_simple[0]
    converter.idx = {"bus": pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 10

    # Act
    converter._create_pgm_input_transformers()

    # Assert
    assert_struct_array_equal(converter.pgm_data["transformer"], pgm_example_simple["transformer"])


def test_create_pgm_input_shunts(
    pp_example_simple: Tuple[PandasData, Dict[str, float]], pgm_example_simple: SingleDataset
):
    # Arrange
    converter = PandaPowerConverter(grid_config=pp_example_simple[1])
    converter.pp_data = pp_example_simple[0]
    converter.idx = {"bus": pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 11

    # Act
    converter._create_pgm_input_shunts()

    # Assert
    assert_struct_array_equal(converter.pgm_data["shunt"], pgm_example_simple["shunt"])


def test_create_pgm_input_sym_gens(
    pp_example_simple: Tuple[PandasData, Dict[str, float]], pgm_example_simple: SingleDataset
):
    # Arrange
    converter = PandaPowerConverter(grid_config=pp_example_simple[1])
    converter.pp_data = pp_example_simple[0]
    converter.idx = {"bus": pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 12

    # Act
    converter._create_pgm_input_sym_gens()

    # Assert
    assert_struct_array_equal(converter.pgm_data["sym_gen"], pgm_example_simple["sym_gen"])


def test_create_pgm_input_three_winding_transformers(
    pp_example_simple: Tuple[PandasData, Dict[str, float]], pgm_example_simple: SingleDataset
):
    # Arrange
    converter = PandaPowerConverter(grid_config=pp_example_simple[1])
    converter.pp_data = pp_example_simple[0]
    converter.idx = {"bus": pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 13

    # Act
    converter._create_pgm_input_three_winding_transformers()

    # Assert
    assert_struct_array_equal(
        converter.pgm_data["three_winding_transformer"], pgm_example_simple["three_winding_transformer"]
    )


def test_create_pgm_input_links(
    pp_example_simple: Tuple[PandasData, Dict[str, float]], pgm_example_simple: SingleDataset
):
    # Arrange
    converter = PandaPowerConverter(grid_config=pp_example_simple[1])
    converter.pp_data = pp_example_simple[0]
    converter.idx = {"bus": pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 14

    # Act
    converter._create_pgm_input_links()

    # Assert
    assert_struct_array_equal(converter.pgm_data["link"], pgm_example_simple["link"])


def test_get_index__key_error():
    # Arrange
    converter = PandaPowerConverter()

    # Act / Assert
    with pytest.raises(KeyError, match=r"index.*bus"):
        converter._get_ids(key="bus", pp_idx=pd.Series())


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
