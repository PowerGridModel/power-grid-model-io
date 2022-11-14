# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

import numpy as np
import pandapower as pp
import pandas as pd
import pytest
from power_grid_model.data_types import SingleDataset
from power_grid_model.utils import import_input_data

from power_grid_model_io.converters.pandapower_converter import PandaPowerConverter, PandasData

from ...utils import assert_struct_array_equal

DATA_DIR = Path(__file__).parent.parent.parent / "data"


@pytest.fixture
def pp_example_simple() -> PandasData:
    #  (ext #1)    shunt - [104]  - 3w - [105] - sym_gen
    #   |                            |
    #  [101] -OO- [102] ----/----- [103]
    #   |                            |
    #  -/-                        (load #31)
    #   |
    #  [106]

    net = pp.create_empty_network()
    pp.create_bus(net, index=101, vn_kv=110)
    pp.create_bus(net, index=102, vn_kv=20)
    pp.create_bus(net, index=103, vn_kv=20)
    pp.create_bus(net, index=104, vn_kv=30.1)
    pp.create_bus(net, index=105, vn_kv=60)
    pp.create_bus(net, index=106, vn_kv=34)
    pp.create_ext_grid(net, index=1, in_service=True, bus=101, vm_pu=31.02)
    pp.create_transformer_from_parameters(
        net,
        index=101,
        hv_bus=101,
        lv_bus=102,
        i0_percent=0.038,
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
    )
    pp.create_line(net, index=101, from_bus=103, to_bus=102, length_km=1.23, std_type="NAYY 4x150 SE")
    pp.create_load(
        net, index=101, bus=103, p_mw=2.5, q_mvar=0.24, const_i_percent=26.0, const_z_percent=51.0, cos_phi=2
    )
    pp.create_switch(net, index=101, et="l", bus=103, element=101, closed=False)
    pp.create_switch(net, index=3021, et="b", bus=101, element=106, closed=True)
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

    return {component: net[component] for component in net if isinstance(net[component], pd.DataFrame)}


@pytest.fixture
def pgm_example_simple() -> SingleDataset:
    return import_input_data(DATA_DIR / "pandapower.json")


def test_create_pgm_input_nodes(pp_example_simple: PandasData, pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_data = pp_example_simple

    # Act
    converter._create_pgm_input_nodes()

    # Assert
    np.testing.assert_array_equal(converter.pgm_data["node"], pgm_example_simple["node"])


def test_create_pgm_input_lines(pp_example_simple: PandasData, pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_data = pp_example_simple
    converter.idx = {"bus": pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 5

    # Act
    converter._create_pgm_input_lines()

    # Assert
    assert_struct_array_equal(converter.pgm_data["line"], pgm_example_simple["line"])


def test_create_pgm_input_sources(pp_example_simple: PandasData, pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_data = pp_example_simple
    converter.idx = {"bus": pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 6

    # Act
    converter._create_pgm_input_sources()

    # Assert
    assert_struct_array_equal(converter.pgm_data["source"], pgm_example_simple["source"])


def test_create_pgm_input_sym_loads(pp_example_simple: PandasData, pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_data = pp_example_simple
    converter.idx = {"bus": pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 7

    # Act
    converter._create_pgm_input_sym_loads()

    # Assert
    assert_struct_array_equal(converter.pgm_data["sym_load"], pgm_example_simple["sym_load"])


def test_create_pgm_input_transformers(pp_example_simple: PandasData, pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_data = pp_example_simple
    converter.idx = {"bus": pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 10

    # Act
    converter._create_pgm_input_transformers()

    # Assert
    assert_struct_array_equal(converter.pgm_data["transformer"], pgm_example_simple["transformer"])


def test_create_pgm_input_shunts(pp_example_simple: PandasData, pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_data = pp_example_simple
    converter.idx = {"bus": pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 11

    # Act
    converter._create_pgm_input_shunts()

    # Assert
    assert_struct_array_equal(converter.pgm_data["shunt"], pgm_example_simple["shunt"])


def test_create_pgm_input_sym_gens(pp_example_simple: PandasData, pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_data = pp_example_simple
    converter.idx = {"bus": pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 12

    # Act
    converter._create_pgm_input_sym_gens()

    # Assert
    assert_struct_array_equal(converter.pgm_data["sym_gen"], pgm_example_simple["sym_gen"])


def test_create_pgm_input_three_winding_transformers(pp_example_simple: PandasData, pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_data = pp_example_simple
    converter.idx = {"bus": pd.Series([0, 1, 2, 3, 4, 5], index=[101, 102, 103, 104, 105, 106], dtype=np.int32)}
    converter.next_idx = 13

    # Act
    converter._create_pgm_input_three_winding_transformers()

    # Assert
    assert_struct_array_equal(
        converter.pgm_data["three_winding_transformer"], pgm_example_simple["three_winding_transformer"]
    )


def test_create_pgm_input_links(pp_example_simple: PandasData, pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_data = pp_example_simple
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


def test_get_load_p_specified():
    # Arrange
    pp_loads = pd.DataFrame(
        [[20, 10.5, 400.0, 32.1], [62.0, 10.0, 400.0, 32.1], [None, 15.2, 400.0, 32.1]],
        columns=["p_mw", "scaling", "sn_mva", "cos_phi"],
    )
    expected_value = np.array([672.0, 1984.0, None], dtype=np.float64)  # third is wrong

    # Act
    actual_value = PandaPowerConverter._get_load_p_specified(3.2, pp_loads)

    # Assert
    np.testing.assert_array_equal(actual_value, expected_value)


# def test_get_load_q_specified():
#     # Arrange
#     pp_loads = pd.DataFrame(
#         [[20, 10.5, 400.0, 32.1], [62.0, 10.0, 400.0, 32.1], [None, 15.2, 400.0, 32.1]],
#         columns=["q_mva", "scaling", "sn_mva", "cos_phi"],
#     )
#     expected_value = np.array([672.0, 1984.0, None], dtype=np.float64)  # third is wrong
#
#     # Act
#     actual_value = PandaPowerConverter._get_load_q_specified(3.2, pp_loads)
#
#     # Assert
#     np.testing.assert_array_equal(actual_value, expected_value)
