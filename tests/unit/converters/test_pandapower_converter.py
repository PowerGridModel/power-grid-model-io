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
    #  (ext #1)
    #   |
    #  [11] -OO- [12] ----/- [13]
    #                          |
    #                     (load #31)
    net = pp.create_empty_network()
    pp.create_bus(net, index=101, vn_kv=110)
    pp.create_bus(net, index=102, vn_kv=20)
    pp.create_bus(net, index=103, vn_kv=20)
    pp.create_ext_grid(net, index=1, in_service=True, bus=101, vm_pu=31.02)
    # pp.create_transformer_from_parameters(
    #     net,
    #     index=101,
    #     hv_bus=101,
    #     lv_bus=102,
    #     i0_percent=0.038,
    #     pfe_kw=11.6,
    #     vkr_percent=0.322,
    #     sn_mva=40,
    #     vn_lv_kv=22.0,
    #     vn_hv_kv=110.0,
    #     vk_percent=17.8,
    # )
    pp.create_line(net, index=101, from_bus=103, to_bus=102, length_km=1.23, std_type="NAYY 4x150 SE")
    pp.create_load(net, index=101, bus=103, p_mw=2.5, q_mvar=0.24, const_i_percent=26.0, const_z_percent=51.0)
    pp.create_switch(net, index=101, et="l", bus=103, element=101, closed=False)

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
    pd.testing.assert_series_equal(converter.idx["bus"], pd.Series([0, 1, 2], index=[101, 102, 103], dtype=np.int32))


def test_create_pgm_input_lines(pp_example_simple: PandasData, pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_data = pp_example_simple
    converter.idx = {"bus": pd.Series([0, 1, 2], index=[101, 102, 103], dtype=np.int32)}
    converter.next_idx = 3

    # Act
    converter._create_pgm_input_lines()

    # Assert
    assert_struct_array_equal(converter.pgm_data["line"], pgm_example_simple["line"])
    pd.testing.assert_series_equal(converter.idx["line"], pd.Series([3], index=[101], dtype=np.int32))


def test_create_pgm_input_sources(pp_example_simple: PandasData, pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_data = pp_example_simple
    converter.idx = {"bus": pd.Series([0, 1, 2], index=[101, 102, 103], dtype=np.int32)}
    converter.next_idx = 4

    # Act
    converter._create_pgm_input_sources()

    # Assert
    assert_struct_array_equal(converter.pgm_data["source"], pgm_example_simple["source"])
    # pd.testing.assert_series_equal(converter.idx["ext_grid"], pd.Series([4], index=[1], dtype=np.int32))


def test_create_pgm_input_sym_loads(pp_example_simple: PandasData, pgm_example_simple: SingleDataset):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_data = pp_example_simple
    converter.idx = {"bus": pd.Series([0, 1, 2], index=[101, 102, 103], dtype=np.int32)}
    converter.next_idx = 5

    # Act
    converter._create_pgm_input_sym_loads()

    # Assert
    assert_struct_array_equal(converter.pgm_data["sym_load"], pgm_example_simple["sym_load"])


# def test_create_pgm_input_shunt(pp_example_simple: PandasData, pgm_example_simple: SingleDataset):
#     # Arrange
#     converter = PandaPowerConverter()
#     converter.pp_data = pp_example_simple
#
#     # Act
#     converter._create_pgm_input_shunt()
#
#     # Assert
#     np.testing.assert_array_equal(converter.pgm_data["shunt"], pgm_example_simple["shunt"])
#     # pd.testing.assert_series_equal(converter.idx["bus"], pd.Series([0, 1, 2], index=[101, 102, 103], dtype=np.int32))


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
