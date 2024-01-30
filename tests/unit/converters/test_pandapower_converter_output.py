# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Callable, List
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from power_grid_model import initialize_array

from power_grid_model_io.converters.pandapower_converter import PandaPowerConverter
from tests.utils import MockDf


@pytest.fixture
def converter() -> PandaPowerConverter:
    converter = PandaPowerConverter()
    converter._get_pp_ids = MagicMock()  # type: ignore
    converter.pp_output_data = MagicMock()
    return converter


def test_create_output_data():
    # Arrange
    converter = MagicMock()

    # Act
    PandaPowerConverter._create_output_data(self=converter)  # type: ignore

    # Assert
    assert len(converter.method_calls) == 13
    converter._pp_buses_output.assert_called_once_with()
    converter._pp_lines_output.assert_called_once_with()
    converter._pp_ext_grids_output.assert_called_once_with()
    converter._pp_shunts_output.assert_called_once_with()
    converter._pp_sgens_output.assert_called_once_with()
    converter._pp_trafos_output.assert_called_once_with()
    converter._pp_trafos3w_output.assert_called_once_with()
    converter._pp_loads_output.assert_called_once_with()
    converter._pp_asym_loads_output.assert_called_once_with()
    converter._pp_asym_gens_output.assert_called_once_with()
    converter._pp_motor_output.assert_called_once_with()
    converter._pp_ward_output.assert_called_once_with()
    converter._pp_switches_output.assert_called_once_with()


def test_create_output_data_3ph():
    # Arrange
    converter = MagicMock()

    # Act
    PandaPowerConverter._create_output_data_3ph(self=converter)  # type: ignore

    # Assert
    assert len(converter.method_calls) == 8
    converter._pp_buses_output_3ph.assert_called_once_with()
    converter._pp_lines_output_3ph.assert_called_once_with()
    converter._pp_ext_grids_output_3ph.assert_called_once_with()
    converter._pp_sgens_output_3ph.assert_called_once_with()
    converter._pp_trafos_output_3ph.assert_called_once_with()
    converter._pp_loads_output_3ph.assert_called_once_with()
    converter._pp_asym_loads_output_3ph.assert_called_once_with()
    converter._pp_asym_gens_output_3ph.assert_called_once_with()


@pytest.mark.parametrize(
    ("create_fn", "table"),
    [
        (PandaPowerConverter._pp_buses_output, "node"),
        (PandaPowerConverter._pp_lines_output, "line"),
        (PandaPowerConverter._pp_ext_grids_output, "source"),
        (PandaPowerConverter._pp_shunts_output, "shunt"),
        (PandaPowerConverter._pp_sgens_output, "sym_gen"),
        (PandaPowerConverter._pp_trafos_output, "transformer"),
        (PandaPowerConverter._pp_trafos3w_output, "three_winding_transformer"),
        (PandaPowerConverter._pp_loads_output, "sym_load"),
        (PandaPowerConverter._pp_asym_loads_output, "asym_load"),
        (PandaPowerConverter._pp_asym_gens_output, "asym_gen"),
        (PandaPowerConverter._pp_ward_output, "ward"),
        (PandaPowerConverter._pp_motor_output, "motor"),
        (PandaPowerConverter._pp_switches_output, "link"),
        (PandaPowerConverter._pp_buses_output_3ph, "node"),
        (PandaPowerConverter._pp_lines_output_3ph, "line"),
        (PandaPowerConverter._pp_ext_grids_output_3ph, "source"),
        (PandaPowerConverter._pp_sgens_output_3ph, "sym_gen"),
        (PandaPowerConverter._pp_trafos_output_3ph, "transformer"),
        (PandaPowerConverter._pp_loads_output_3ph, "sym_load"),
        (PandaPowerConverter._pp_asym_loads_output_3ph, "asym_load"),
        (PandaPowerConverter._pp_asym_gens_output_3ph, "asym_gen"),
    ],
)
def test_create_pp_output_object__empty(create_fn: Callable[[PandaPowerConverter], None], table: str):
    # Arrange: No table
    converter = PandaPowerConverter()

    # Act / Assert
    with patch("power_grid_model_io.converters.pandapower_converter.pd.DataFrame") as mock_df:
        create_fn(converter)
        mock_df.assert_not_called()

    # Arrange: Empty table
    converter.pgm_output_data[table] = np.array([])  # type: ignore

    # Act / Assert
    with patch("power_grid_model_io.converters.pandapower_converter.pd.DataFrame") as mock_df:
        create_fn(converter)
        mock_df.assert_not_called()


def test_output_bus(converter):
    # Arrange
    mock_pgm_array = MagicMock()
    converter.pgm_output_data["node"] = mock_pgm_array

    with patch("power_grid_model_io.converters.pandapower_converter.pd.DataFrame") as mock_pp_df:
        # Act
        converter._pp_buses_output()

        # Assert

        # initialization
        converter._get_pp_ids.assert_called_once_with("bus", mock_pgm_array["X"])

        # retrieval
        mock_pgm_array.__getitem__.assert_any_call("id")
        mock_pgm_array.__getitem__.assert_any_call("u_pu")
        mock_pgm_array.__getitem__.assert_any_call("u_angle")

        # assignment
        mock_pp_df.return_value.__setitem__.assert_any_call("vm_pu", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("va_degree", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_mvar", ANY)

        # result
        converter.pp_output_data.__setitem__.assert_called_once_with("res_bus", mock_pp_df.return_value)


def test_output_line(converter):
    # Arrange
    mock_pgm_array = MagicMock()
    converter.pgm_nodes_lookup = MagicMock()
    converter.pgm_output_data["line"] = mock_pgm_array
    converter.pgm_input_data["line"] = MagicMock()

    with patch("power_grid_model_io.converters.pandapower_converter.pd.DataFrame") as mock_pp_df:
        # Act
        converter._pp_lines_output()

        # Assert

        # initialization
        converter._get_pp_ids.assert_called_once_with("line", mock_pgm_array["id"])

        # retrieval
        mock_pgm_array.__getitem__.assert_any_call("id")
        mock_pgm_array.__getitem__.assert_any_call("p_from")
        mock_pgm_array.__getitem__.assert_any_call("q_from")
        mock_pgm_array.__getitem__.assert_any_call("p_to")
        mock_pgm_array.__getitem__.assert_any_call("q_to")
        mock_pgm_array.__getitem__.assert_any_call("i_from")
        mock_pgm_array.__getitem__.assert_any_call("i_to")
        mock_pgm_array.__getitem__.assert_any_call("loading")

        # assignment
        mock_pp_df.return_value.__setitem__.assert_any_call("p_from_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_from_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_to_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_to_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("pl_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("ql_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_from_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_to_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("loading_percent", ANY)

        # result
        converter.pp_output_data.__setitem__.assert_called_once_with("res_line", mock_pp_df.return_value)


def test_output_line__node_lookup():
    # Arrange
    converter = PandaPowerConverter()
    converter.idx_lookup[("line", None)] = pd.Series([132, 121], index=[32, 21])
    converter.pgm_input_data["line"] = initialize_array("input", "line", 2)
    converter.pgm_input_data["line"]["id"] = [21, 32]
    converter.pgm_input_data["line"]["from_node"] = [2, 3]
    converter.pgm_input_data["line"]["to_node"] = [1, 2]
    converter.pgm_output_data["line"] = initialize_array("sym_output", "line", 2)
    converter.pgm_output_data["line"]["id"] = [21, 32]
    converter.pgm_nodes_lookup = pd.DataFrame({"u_pu": [1.1, 1.2, 1.3], "u_degree": [0.1, 0.2, 0.3]}, index=[1, 2, 3])

    # Act
    converter._pp_lines_output()

    # Assert
    pp_output_lines = converter.pp_output_data["res_line"]
    pd.testing.assert_series_equal(
        pp_output_lines["vm_from_pu"],
        pd.Series([1.2, 1.3], name="vm_from_pu", index=[121, 132]),
    )
    pd.testing.assert_series_equal(
        pp_output_lines["vm_to_pu"],
        pd.Series([1.1, 1.2], name="vm_to_pu", index=[121, 132]),
    )
    pd.testing.assert_series_equal(
        pp_output_lines["va_from_degree"],
        pd.Series([0.2, 0.3], name="va_from_degree", index=[121, 132]),
    )
    pd.testing.assert_series_equal(
        pp_output_lines["va_to_degree"],
        pd.Series([0.1, 0.2], name="va_to_degree", index=[121, 132]),
    )


def test_output_line__node_lookup__exception(converter):
    # Arrange
    converter.pgm_nodes_lookup = MagicMock()
    converter.pgm_output_data["line"] = initialize_array("input", "line", 3)
    converter.pgm_input_data["line"] = initialize_array("sym_output", "line", 3)
    converter.pgm_output_data["line"]["id"] = [1, 2, 3]
    converter.pgm_input_data["line"]["id"] = [1, 3, 2]

    # Act / Assert
    with pytest.raises(ValueError, match="The output line ids should correspond to the input line ids"):
        converter._pp_lines_output()


def test_output_ext_grids(converter):
    # Arrange
    mock_pgm_array = MagicMock()
    converter.pgm_output_data["source"] = mock_pgm_array

    with patch("power_grid_model_io.converters.pandapower_converter.pd.DataFrame") as mock_pp_df:
        # Act
        converter._pp_ext_grids_output()

        # Assert

        # initialization
        converter._get_pp_ids.assert_called_once_with("ext_grid", mock_pgm_array["X"])

        # retrieval
        mock_pgm_array.__getitem__.assert_any_call("id")
        mock_pgm_array.__getitem__.assert_any_call("p")
        mock_pgm_array.__getitem__.assert_any_call("q")

        # assignment
        mock_pp_df.return_value.__setitem__.assert_any_call("p_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_mvar", ANY)

        # result
        converter.pp_output_data.__setitem__.assert_called_once_with("res_ext_grid", mock_pp_df.return_value)


def test_output_shunts(converter):
    # Arrange
    mock_pgm_array = MagicMock()
    converter.pgm_input_data["shunt"] = mock_pgm_array
    converter.pgm_output_data["shunt"] = mock_pgm_array
    converter.pgm_nodes_lookup = pd.DataFrame({"u_pu": mock_pgm_array}, index=mock_pgm_array)

    with patch("power_grid_model_io.converters.pandapower_converter.pd.DataFrame") as mock_pp_df:
        # Act
        converter._pp_shunts_output()

        # Assert

        # initialization
        converter._get_pp_ids.assert_called_once_with("shunt", mock_pgm_array["X"])

        # retrieval
        mock_pgm_array.__getitem__.assert_any_call("id")
        mock_pgm_array.__getitem__.assert_any_call("node")
        mock_pgm_array.__getitem__.assert_any_call("p")
        mock_pgm_array.__getitem__.assert_any_call("q")
        # mock_pgm_array.__getitem__.assert_any_call("u_pu")

        # assignment
        mock_pp_df.return_value.__setitem__.assert_any_call("p_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("vm_pu", ANY)

        # result
        converter.pp_output_data.__setitem__.assert_called_once_with("res_shunt", mock_pp_df.return_value)


def test_output_sgen(converter):
    # Arrange
    mock_pgm_array = MagicMock()
    converter.pgm_output_data["sym_gen"] = mock_pgm_array

    with patch("power_grid_model_io.converters.pandapower_converter.pd.DataFrame") as mock_pp_df:
        # Act
        converter._pp_sgens_output()

        # Assert

        # initialization
        converter._get_pp_ids.assert_called_once_with("sgen", mock_pgm_array["X"])

        # retrieval
        mock_pgm_array.__getitem__.assert_any_call("id")
        mock_pgm_array.__getitem__.assert_any_call("p")
        mock_pgm_array.__getitem__.assert_any_call("q")

        # assignment
        mock_pp_df.return_value.__setitem__.assert_any_call("p_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_mvar", ANY)

        # result
        converter.pp_output_data.__setitem__.assert_called_once_with("res_sgen", mock_pp_df.return_value)


def test_output_trafos__current(converter):
    # Arrange
    mock_pgm_array = MagicMock()
    converter.trafo_loading = "current"
    converter.pp_input_data["trafo"] = MockDf(2)
    converter.pgm_input_data["transformer"] = mock_pgm_array
    converter.pgm_output_data["transformer"] = mock_pgm_array
    converter.pgm_nodes_lookup = pd.DataFrame(
        {"u_pu": mock_pgm_array, "u_degree": mock_pgm_array}, index=mock_pgm_array
    )

    with patch("power_grid_model_io.converters.pandapower_converter.pd.DataFrame") as mock_pp_df:
        # Act
        converter._pp_trafos_output()

        # Assert

        # initialization
        converter._get_pp_ids.assert_called_once_with("trafo", mock_pgm_array["X"])

        # retrieval
        mock_pgm_array.__getitem__.assert_any_call("id")
        mock_pgm_array.__getitem__.assert_any_call("from_node")
        mock_pgm_array.__getitem__.assert_any_call("to_node")
        mock_pgm_array.__getitem__.assert_any_call("p_from")
        mock_pgm_array.__getitem__.assert_any_call("q_from")
        mock_pgm_array.__getitem__.assert_any_call("p_to")
        mock_pgm_array.__getitem__.assert_any_call("q_to")
        mock_pgm_array.__getitem__.assert_any_call("i_from")
        mock_pgm_array.__getitem__.assert_any_call("i_to")
        # current loading retrieval
        mock_pgm_array.__getitem__.assert_any_call("sn")

        # assignment
        mock_pp_df.return_value.__setitem__.assert_any_call("p_hv_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_hv_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_lv_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_lv_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("pl_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("ql_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_hv_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_lv_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("vm_hv_pu", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("vm_lv_pu", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("va_hv_degree", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("va_lv_degree", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("loading_percent", ANY)

        # result
        converter.pp_output_data.__setitem__.assert_called_once_with("res_trafo", mock_pp_df.return_value)


def test_output_trafos__power(converter):
    # Arrange
    mock_pgm_array = MagicMock()
    converter.trafo_loading = "power"
    converter.pp_input_data["trafo"] = MockDf(2)
    converter.pgm_input_data["transformer"] = mock_pgm_array
    converter.pgm_output_data["transformer"] = mock_pgm_array
    converter.pgm_nodes_lookup = pd.DataFrame(
        {"u_pu": mock_pgm_array, "u_degree": mock_pgm_array}, index=mock_pgm_array
    )

    with patch("power_grid_model_io.converters.pandapower_converter.pd.DataFrame") as mock_pp_df:
        # Act
        converter._pp_trafos_output()

        mock_pgm_array.__getitem__.assert_any_call("loading")
        mock_pp_df.return_value.__setitem__.assert_any_call("loading_percent", ANY)
        # result
        converter.pp_output_data.__setitem__.assert_called_once_with("res_trafo", mock_pp_df.return_value)


def test_output_trafos__invalid_trafo_loading(converter):
    # Arrange
    mock_pgm_array = MagicMock()
    converter.trafo_loading = "abcd"
    converter.pp_input_data["trafo"] = MockDf(2)
    converter.pgm_input_data["transformer"] = mock_pgm_array
    converter.pgm_output_data["transformer"] = mock_pgm_array
    converter.pgm_nodes_lookup = pd.DataFrame(
        {"u_pu": mock_pgm_array, "u_degree": mock_pgm_array}, index=mock_pgm_array
    )

    with pytest.raises(ValueError, match="Invalid transformer loading type: abcd"):
        converter._pp_trafos_output()


def test_output_trafo3w(converter):
    # Arrange
    mock_pgm_array = MagicMock()
    converter.pgm_input_data["three_winding_transformer"] = mock_pgm_array
    converter.pgm_output_data["three_winding_transformer"] = mock_pgm_array
    converter.pgm_nodes_lookup = pd.DataFrame(
        {"u_pu": mock_pgm_array, "u_degree": mock_pgm_array}, index=mock_pgm_array
    )

    with patch("power_grid_model_io.converters.pandapower_converter.pd.DataFrame") as mock_pp_df:
        # Act
        converter._pp_trafos3w_output()

        # Assert

        # initialization
        converter._get_pp_ids.assert_called_once_with("trafo3w", mock_pgm_array["X"])

        # retrieval
        mock_pgm_array.__getitem__.assert_any_call("id")
        mock_pgm_array.__getitem__.assert_any_call("node_1")
        mock_pgm_array.__getitem__.assert_any_call("node_2")
        mock_pgm_array.__getitem__.assert_any_call("node_3")
        mock_pgm_array.__getitem__.assert_any_call("p_1")
        mock_pgm_array.__getitem__.assert_any_call("q_1")
        mock_pgm_array.__getitem__.assert_any_call("p_2")
        mock_pgm_array.__getitem__.assert_any_call("q_2")
        mock_pgm_array.__getitem__.assert_any_call("p_3")
        mock_pgm_array.__getitem__.assert_any_call("q_3")
        mock_pgm_array.__getitem__.assert_any_call("i_1")
        mock_pgm_array.__getitem__.assert_any_call("i_2")
        mock_pgm_array.__getitem__.assert_any_call("i_3")
        # TODO find a better way for getting voltages at node
        # mock_pgm_array.__getitem__.assert_any_call("u_pu")
        # mock_pgm_array.__getitem__.assert_any_call("u_degree")
        mock_pgm_array.__getitem__.assert_any_call("loading")

        # assignment
        mock_pp_df.return_value.__setitem__.assert_any_call("p_hv_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_hv_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_mv_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_mv_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_lv_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_lv_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("pl_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("ql_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_hv_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_mv_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_lv_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("vm_hv_pu", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("vm_mv_pu", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("vm_lv_pu", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("va_hv_degree", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("va_mv_degree", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("va_lv_degree", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("loading_percent", ANY)

        # result
        converter.pp_output_data.__setitem__.assert_called_once_with("res_trafo3w", mock_pp_df.return_value)


def test_pp_load_result_accumulate__sym():
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
    load_id_names = ["const_power", "const_current", "const_impedance"]

    # Act
    actual = converter._pp_load_result_accumulate(pp_component_name="load", load_id_names=load_id_names)

    # Assert
    pd.testing.assert_frame_equal(actual, expected)


def test_pp_load_result_accumulate__asym():
    # Arrange
    converter = PandaPowerConverter()
    converter.pgm_output_data["sym_load"] = initialize_array("asym_output", "sym_load", 6)
    converter.pgm_output_data["sym_load"]["id"] = [0, 1, 2, 3, 4, 5]
    converter.pgm_output_data["sym_load"]["p"] = [
        [0.5e6, 0.4e6, 0.1e6],
        [1.1e6, 0.8e6, 0.1e6],
        [2.1e6, 1e6, 0.9e6],
        [2e6, 2e6, 4e6],
        [6e6, 5e6, 5e6],
        [12e6, 12e6, 8e6],
    ]
    converter.pgm_output_data["sym_load"]["q"] = [
        [0.3e4, 0.2e4, 0.5e4],
        [0.7e4, 0.7e4, 0.6e4],
        [1.1e4, 1.4e4, 1.5e4],
        [3e4, 4e4, 1e4],
        [6e4, 4e4, 6e4],
        [11e4, 10e4, 11e4],
    ]
    converter.idx[("load", "const_power")] = pd.Series([2, 4], index=[101, 100])
    converter.idx[("load", "const_current")] = pd.Series([1, 3], index=[102, 100])
    converter.idx[("load", "const_impedance")] = pd.Series([0, 5], index=[101, 102])

    expected = pd.DataFrame(
        [[16.0 + 8.0, 0.16 + 0.08], [4.0 + 1.0, 0.04 + 0.01], [2.0 + 32.0, 0.02 + 0.32]],
        columns=["p_mw", "q_mvar"],
        index=[100, 101, 102],
    )
    load_id_names = ["const_power", "const_current", "const_impedance"]

    # Act
    actual = converter._pp_load_result_accumulate(pp_component_name="load", load_id_names=load_id_names)

    # Assert
    pd.testing.assert_frame_equal(actual, expected)


@pytest.mark.parametrize(
    ("output_fn", "table", "load_id_names", "result_suffix"),
    [
        (PandaPowerConverter._pp_loads_output, "load", ["const_power", "const_impedance", "const_current"], ""),
        (PandaPowerConverter._pp_motor_output, "motor", ["motor_load"], ""),
        (PandaPowerConverter._pp_loads_output_3ph, "load", ["const_power", "const_impedance", "const_current"], "_3ph"),
    ],
)
def test_output_load_types(
    output_fn: Callable[[PandaPowerConverter], None], table: str, load_id_names: List[str], result_suffix: str
):
    # Arrange
    converter = PandaPowerConverter()
    converter.pgm_output_data["sym_load"] = initialize_array("sym_output", "sym_load", 6)
    converter.idx[(table, load_id_names[0])] = pd.Series([0], index=[1])
    converter._pp_load_result_accumulate = MagicMock()  # type: ignore

    # Act
    output_fn(converter)

    # Assert
    converter._pp_load_result_accumulate.assert_called_once_with(pp_component_name=table, load_id_names=load_id_names)
    assert converter.pp_output_data["res_" + table + result_suffix] == converter._pp_load_result_accumulate.return_value


def test_output_load_ward():
    # Arrange
    converter = PandaPowerConverter()
    load_id_names = ["ward_const_power_load", "ward_const_impedance_load"]

    converter.pgm_output_data["sym_load"] = initialize_array("sym_output", "sym_load", 6)
    converter.idx[("ward", load_id_names[0])] = pd.Series([0], index=[1])
    converter._pp_load_result_accumulate = MagicMock()

    # Act
    converter._pp_ward_output()

    # Assert
    converter._pp_load_result_accumulate.assert_called_once_with(pp_component_name="ward", load_id_names=load_id_names)
    assert converter.pp_output_data["res_ward"] == converter._pp_load_result_accumulate.return_value


def test_output_asymmetric_load(converter):
    # Arrange
    mock_pgm_array = MagicMock()
    converter.pgm_output_data["asym_load"] = mock_pgm_array

    with patch("power_grid_model_io.converters.pandapower_converter.pd.DataFrame") as mock_pp_df:
        # Act
        converter._pp_asym_loads_output()

        # Assert

        # initialization
        converter._get_pp_ids.assert_called_once_with("asymmetric_load", mock_pgm_array["X"])

        # retrieval
        mock_pgm_array.__getitem__.assert_any_call("id")
        mock_pgm_array.__getitem__.assert_any_call("p")
        mock_pgm_array.__getitem__.assert_any_call("q")

        # assignment
        mock_pp_df.return_value.__setitem__.assert_any_call("p_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_mvar", ANY)

        # result
        converter.pp_output_data.__setitem__.assert_called_once_with("res_asymmetric_load", mock_pp_df.return_value)


def test_output_asymmetric_sgen(converter):
    # Arrange
    mock_pgm_array = MagicMock()
    converter.pgm_output_data["asym_gen"] = mock_pgm_array

    with patch("power_grid_model_io.converters.pandapower_converter.pd.DataFrame") as mock_pp_df:
        # Act
        converter._pp_asym_gens_output()

        # Assert

        # initialization
        converter._get_pp_ids.assert_called_once_with("asymmetric_sgen", mock_pgm_array["X"])

        # retrieval
        mock_pgm_array.__getitem__.assert_any_call("id")
        mock_pgm_array.__getitem__.assert_any_call("p")
        mock_pgm_array.__getitem__.assert_any_call("q")

        # assignment
        mock_pp_df.return_value.__setitem__.assert_any_call("p_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_mvar", ANY)

        # result
        converter.pp_output_data.__setitem__.assert_called_once_with("res_asymmetric_sgen", mock_pp_df.return_value)


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


def test_pp_buses_output__accumulate_power__component_absent():
    # Arrange
    converter = PandaPowerConverter()
    converter.idx_lookup = {("bus", None): pd.Series([101, 102, 103, 104], index=[0, 1, 2, 3], dtype=np.int32)}
    pp_buses = pd.DataFrame(np.empty((4, 2), np.float64), columns=["p_mw", "q_mvar"], index=[101, 102, 103, 104])
    converter.pgm_input_data = {
        "link": initialize_array("input", "link", 2),
    }
    converter.pgm_output_data = {
        "line": initialize_array("sym_output", "line", 3),
    }
    with pytest.raises(KeyError, match="PGM input_data is needed to accumulate output for lines"):
        converter._pp_buses_output__accumulate_power(pp_buses)


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


def test_pp_buses_output__accumulate_power__output_empty():
    # Arrange
    converter = PandaPowerConverter()
    converter.idx_lookup = {("bus", None): pd.Series([101, 102, 103], index=[0, 1, 2], dtype=np.int32)}
    pp_buses = pd.DataFrame(np.empty((3, 2), np.float64), columns=["p_mw", "q_mvar"], index=[101, 102, 103])

    converter.pgm_input_data = {
        "link": initialize_array("input", "link", 2),
    }
    converter.pgm_output_data = {
        "link": initialize_array("sym_output", "link", 0),
    }
    converter.pgm_input_data["link"]["from_node"] = [0, 1]
    converter.pgm_input_data["link"]["to_node"] = [1, 2]

    # Act
    converter._pp_buses_output__accumulate_power(pp_buses)

    # Assert
    assert pp_buses["p_mw"][101] == 0
    assert pp_buses["p_mw"][102] == 0
    assert pp_buses["p_mw"][103] == 0
    assert pp_buses["q_mvar"][101] == 0
    assert pp_buses["q_mvar"][102] == 0
    assert pp_buses["q_mvar"][103] == 0


def test_pp_switch_output():
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {
        "trafo": pd.DataFrame({"hv_bus": [1], "lv_bus": [2]}, index=[10]),
        "trafo3w": pd.DataFrame({"hv_bus": [1], "mv_bus": [4], "lv_bus": [5]}, index=[10]),
        "line": pd.DataFrame({"from_bus": [3], "to_bus": [6]}, index=[11]),
        "switch": pd.DataFrame(
            {
                "bus": [1, 2, 1, 4, 5, 3, 6, 7, 8],
                "element": [10, 10, 10, 10, 10, 11, 11, 77, 88],
                "et": ["t", "t", "t3", "t3", "t3", "l", "l", "b", "b"],
                "closed": [True, True, True, True, True, True, True, True, True],
            },
            index=[40, 41, 42, 43, 44, 45, 46, 47, 48],
        ),
    }
    converter.pp_output_data = {
        "res_trafo": pd.DataFrame({"i_hv_ka": [11.1], "i_lv_ka": [11.2]}, index=[10]),
        "res_trafo3w": pd.DataFrame({"i_hv_ka": [12.1], "i_mv_ka": [12.2], "i_lv_ka": [12.3]}, index=[10]),
        "res_line": pd.DataFrame({"i_from_ka": [13.1], "i_to_ka": [13.2]}, index=[11]),
    }
    converter.pgm_output_data = {"link": initialize_array("sym_output", "link", 2)}
    converter.pgm_output_data["link"]["id"] = [101, 102]
    converter.pgm_output_data["link"]["i_from"] = [14100, 14200]
    converter.pgm_output_data["link"]["i_to"] = [15.1, 15.2]

    converter.idx_lookup = {("switch", "b2b_switches"): pd.Series([47, 48], index=[101, 102], dtype=np.int32)}

    expected = pd.DataFrame(
        {
            "i_ka": [11.1, 11.2, 12.1, 12.2, 12.3, 13.1, 13.2, 14.1, 14.2],
            "loading_percent": [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        },
        index=[40, 41, 42, 43, 44, 45, 46, 47, 48],
    )

    # Act
    converter._pp_switches_output()
    actual = converter.pp_output_data["res_switch"]

    # Assert
    pd.testing.assert_frame_equal(actual, expected, check_dtype=False)


@patch("power_grid_model_io.converters.pandapower_converter.np.matmul")
def test_output_bus_3ph(mock_np_array: MagicMock, converter):
    # Arrange
    mock_pgm_array = MagicMock()
    converter.pgm_output_data["node"] = mock_pgm_array
    converter._pp_buses_output_3ph__accumulate_power = MagicMock()

    with patch("power_grid_model_io.converters.pandapower_converter.pd.DataFrame") as mock_pp_df:
        # Act
        converter._pp_buses_output_3ph()

        # Assert

        # initialization
        converter._get_pp_ids.assert_called_once_with("bus", mock_pgm_array["X"])

        # retrieval
        mock_pgm_array.__getitem__.assert_any_call("id")
        mock_pgm_array.__getitem__.assert_any_call("u_pu")
        mock_pgm_array.__getitem__.assert_any_call("u_angle")

        # assignment
        mock_pp_df.return_value.__setitem__.assert_any_call("vm_a_pu", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("va_a_degree", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("vm_b_pu", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("va_b_degree", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("vm_c_pu", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("va_c_degree", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("unbalance_percent", ANY)

        converter._pp_buses_output_3ph__accumulate_power.assert_called_once()

        # result
        converter.pp_output_data.__setitem__.assert_called_once_with("res_bus_3ph", mock_pp_df.return_value)


def test_output_line_3ph(converter):
    # Arrange
    mock_pgm_array = MagicMock()
    converter.pgm_nodes_lookup = MagicMock()
    converter.pgm_output_data["line"] = mock_pgm_array
    converter.pgm_output_data["node"] = mock_pgm_array
    converter.pgm_input_data["line"] = MagicMock()

    with patch("power_grid_model_io.converters.pandapower_converter.pd.DataFrame") as mock_pp_df:
        # Act
        converter._pp_lines_output_3ph()

        # Assert

        # initialization
        converter._get_pp_ids.assert_called_once_with("line", mock_pgm_array["X"])

        # retrieval
        mock_pgm_array.__getitem__.assert_any_call("id")
        mock_pgm_array.__getitem__.assert_any_call("p_from")
        mock_pgm_array.__getitem__.assert_any_call("q_from")
        mock_pgm_array.__getitem__.assert_any_call("p_to")
        mock_pgm_array.__getitem__.assert_any_call("q_to")
        mock_pgm_array.__getitem__.assert_any_call("i_from")
        mock_pgm_array.__getitem__.assert_any_call("i_to")
        mock_pgm_array.__getitem__.assert_any_call("loading")

        # assignment
        mock_pp_df.return_value.__setitem__.assert_any_call("p_a_from_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_a_from_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_a_to_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_a_to_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_b_from_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_b_from_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_b_to_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_b_to_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_c_from_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_c_from_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_c_to_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_c_to_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_a_l_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_b_l_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_c_l_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_a_l_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_b_l_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_c_l_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_a_from_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_b_from_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_c_from_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_a_to_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_b_to_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_c_to_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_n_from_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_n_to_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_n_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_a_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_b_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_c_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("loading_percent", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("loading_a_percent", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("loading_b_percent", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("loading_c_percent", ANY)

        # result
        converter.pp_output_data.__setitem__.assert_called_once_with("res_line_3ph", mock_pp_df.return_value)


def test_output_ext_grids_3ph(converter):
    # Arrange
    mock_pgm_array = MagicMock()
    converter.pgm_output_data["source"] = mock_pgm_array

    with patch("power_grid_model_io.converters.pandapower_converter.pd.DataFrame") as mock_pp_df:
        # Act
        converter._pp_ext_grids_output_3ph()

        # Assert

        # initialization
        converter._get_pp_ids.assert_called_once_with("ext_grid", mock_pgm_array["X"])

        # retrieval
        mock_pgm_array.__getitem__.assert_any_call("id")
        mock_pgm_array.__getitem__.assert_any_call("p")
        mock_pgm_array.__getitem__.assert_any_call("q")

        # assignment
        mock_pp_df.return_value.__setitem__.assert_any_call("p_a_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_a_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_b_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_b_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_c_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_c_mvar", ANY)

        # result
        converter.pp_output_data.__setitem__.assert_called_once_with("res_ext_grid_3ph", mock_pp_df.return_value)


def test_output_sgen_3ph(converter):
    # Arrange
    mock_pgm_array = MagicMock()
    converter.pgm_output_data["sym_gen"] = mock_pgm_array

    with patch("power_grid_model_io.converters.pandapower_converter.pd.DataFrame") as mock_pp_df:
        # Act
        converter._pp_sgens_output_3ph()

        # initialization
        converter._get_pp_ids.assert_called_once_with("sgen", mock_pgm_array["X"])

        # retrieval
        mock_pgm_array.__getitem__.assert_any_call("id")
        mock_pgm_array.__getitem__.assert_any_call("p")
        mock_pgm_array.__getitem__.assert_any_call("q")

        # assignment
        mock_pp_df.return_value.__setitem__.assert_any_call("p_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_mvar", ANY)

        # result
        converter.pp_output_data.__setitem__.assert_called_once_with("res_sgen_3ph", mock_pp_df.return_value)


def test_output_trafos_3ph__current(converter):
    # Arrange
    mock_pgm_array = MagicMock()
    converter.trafo_loading = "current"
    converter.pp_input_data["trafo"] = MockDf(2)
    converter.pgm_input_data["transformer"] = mock_pgm_array
    converter.pgm_output_data["transformer"] = mock_pgm_array

    with patch("power_grid_model_io.converters.pandapower_converter.pd.DataFrame") as mock_pp_df:
        # Act
        converter._pp_trafos_output_3ph()

        # Assert

        # initialization
        converter._get_pp_ids.assert_called_once_with("trafo", mock_pgm_array["X"])

        # retrieval
        mock_pgm_array.__getitem__.assert_any_call("id")
        mock_pgm_array.__getitem__.assert_any_call("p_from")
        mock_pgm_array.__getitem__.assert_any_call("q_from")
        mock_pgm_array.__getitem__.assert_any_call("p_to")
        mock_pgm_array.__getitem__.assert_any_call("q_to")
        mock_pgm_array.__getitem__.assert_any_call("i_from")
        mock_pgm_array.__getitem__.assert_any_call("i_to")

        # assignment
        mock_pp_df.return_value.__setitem__.assert_any_call("p_a_hv_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_a_hv_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_a_lv_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_a_lv_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_b_hv_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_b_hv_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_b_lv_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_b_lv_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_c_hv_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_c_hv_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_c_lv_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_c_lv_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_a_l_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_b_l_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_c_l_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_a_l_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_b_l_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_c_l_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_a_hv_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_b_hv_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_c_hv_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_a_lv_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_b_lv_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("i_c_lv_ka", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("loading_a_percent", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("loading_b_percent", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("loading_c_percent", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("loading_percent", ANY)

        # result
        converter.pp_output_data.__setitem__.assert_called_once_with("res_trafo_3ph", mock_pp_df.return_value)


def test_output_trafos_3ph__power(converter):
    # Arrange
    mock_pgm_array = MagicMock()
    converter.trafo_loading = "power"
    converter.pp_input_data["trafo"] = MockDf(2)
    converter.pgm_input_data["transformer"] = mock_pgm_array
    converter.pgm_output_data["transformer"] = mock_pgm_array

    with patch("power_grid_model_io.converters.pandapower_converter.pd.DataFrame") as mock_pp_df:
        # Act
        converter._pp_trafos_output_3ph()

        # retrieval
        mock_pgm_array.__getitem__.assert_any_call("loading")

        # result
        converter.pp_output_data.__setitem__.assert_called_once_with("res_trafo_3ph", mock_pp_df.return_value)


def test_output_trafos_3ph__invalid_trafo_loading(converter):
    # Arrange
    mock_pgm_array = MagicMock()
    converter.trafo_loading = "abcd"
    converter.pp_input_data["trafo"] = MockDf(2)
    converter.pgm_input_data["transformer"] = mock_pgm_array
    converter.pgm_output_data["transformer"] = mock_pgm_array

    with pytest.raises(ValueError, match="Invalid transformer loading type: abcd"):
        converter._pp_trafos_output_3ph()


def test_output_asymmetric_load_3ph(converter):
    # Arrange
    mock_pgm_array = MagicMock()
    converter.pgm_output_data["asym_load"] = mock_pgm_array

    with patch("power_grid_model_io.converters.pandapower_converter.pd.DataFrame") as mock_pp_df:
        # Act
        converter._pp_asym_loads_output_3ph()

        # Assert

        # initialization
        converter._get_pp_ids.assert_called_once_with("asymmetric_load", mock_pgm_array["X"])

        # retrieval
        mock_pgm_array.__getitem__.assert_any_call("id")
        mock_pgm_array.__getitem__.assert_any_call("p")
        mock_pgm_array.__getitem__.assert_any_call("q")

        # assignment
        mock_pp_df.return_value.__setitem__.assert_any_call("p_a_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_a_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_b_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_b_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_c_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_c_mvar", ANY)

        # result
        converter.pp_output_data.__setitem__.assert_called_once_with("res_asymmetric_load_3ph", mock_pp_df.return_value)


def test_output_asymmetric_sgen_3ph(converter):
    # Arrange
    mock_pgm_array = MagicMock()
    converter.pgm_output_data["asym_gen"] = mock_pgm_array

    with patch("power_grid_model_io.converters.pandapower_converter.pd.DataFrame") as mock_pp_df:
        # Act
        converter._pp_asym_gens_output_3ph()

        # Assert

        # initialization
        converter._get_pp_ids.assert_called_once_with("asymmetric_sgen", mock_pgm_array["X"])

        # retrieval
        mock_pgm_array.__getitem__.assert_any_call("id")
        mock_pgm_array.__getitem__.assert_any_call("p")
        mock_pgm_array.__getitem__.assert_any_call("q")

        # assignment
        mock_pp_df.return_value.__setitem__.assert_any_call("p_a_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_a_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_b_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_b_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_c_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_c_mvar", ANY)

        # result
        converter.pp_output_data.__setitem__.assert_called_once_with("res_asymmetric_sgen_3ph", mock_pp_df.return_value)


def test_pp_buses_output_3ph__accumulate_power__zero():
    # Arrange
    power_columns = ["p_a_mw", "p_b_mw", "p_c_mw", "q_a_mvar", "q_b_mvar", "q_c_mvar"]
    converter = PandaPowerConverter()
    converter.idx = {"bus": pd.Series([0, 1, 2, 3], index=[101, 102, 103, 104], dtype=np.int32)}
    pp_buses_3ph = pd.DataFrame(np.empty((4, 6), np.float64), columns=power_columns, index=[101, 102, 103, 104])

    expected = pd.DataFrame(
        np.full((4, 6), fill_value=0.0, dtype=np.float64), columns=power_columns, index=[101, 102, 103, 104]
    )

    # Act
    converter._pp_buses_output_3ph__accumulate_power(pp_buses_3ph)

    # Assert
    pd.testing.assert_frame_equal(pp_buses_3ph[power_columns], expected)


def test_pp_buses_output_3ph__accumulate_power__component_absent():
    # Arrange
    power_columns = ["p_a_mw", "p_b_mw", "p_c_mw", "q_a_mvar", "q_b_mvar", "q_c_mvar"]
    converter = PandaPowerConverter()
    converter.idx_lookup = {("bus", None): pd.Series([101, 102, 103, 104], index=[0, 1, 2, 3], dtype=np.int32)}
    pp_buses_3ph = pd.DataFrame(np.empty((4, 6), np.float64), columns=power_columns, index=[101, 102, 103, 104])
    converter.pgm_input_data = {
        "link": initialize_array("input", "link", 2),
    }
    converter.pgm_output_data = {
        "line": initialize_array("asym_output", "line", 3),
    }
    with pytest.raises(KeyError, match="PGM input_data is needed to accumulate output for lines"):
        converter._pp_buses_output_3ph__accumulate_power(pp_buses_3ph)


def test_pp_buses_output_3ph__accumulate_power():
    # We make phase B = 2 x phase A and phase C = 3 x phase A of symmetric test for simplicity in calculations
    multiplier = np.array([1, 2, 3])
    # Arrange
    power_columns = ["p_a_mw", "p_b_mw", "p_c_mw", "q_a_mvar", "q_b_mvar", "q_c_mvar"]
    converter = PandaPowerConverter()
    converter.idx_lookup = {("bus", None): pd.Series([101, 102, 103, 104], index=[0, 1, 2, 3], dtype=np.int32)}
    pp_buses_3ph = pd.DataFrame(np.empty((4, 6), np.float64), columns=power_columns, index=[101, 102, 103, 104])

    converter.pgm_input_data = {
        "line": initialize_array("input", "line", 3),
        "link": initialize_array("input", "link", 2),
        "transformer": initialize_array("input", "transformer", 2),
    }
    converter.pgm_output_data = {
        "line": initialize_array("asym_output", "line", 3),
        "link": initialize_array("asym_output", "link", 2),
        "transformer": initialize_array("asym_output", "transformer", 2),
    }
    converter.pgm_input_data["line"]["from_node"] = [0, 1, 1]
    converter.pgm_input_data["line"]["to_node"] = [1, 2, 3]
    converter.pgm_input_data["link"]["from_node"] = [0, 1]
    converter.pgm_input_data["link"]["to_node"] = [1, 2]
    converter.pgm_input_data["transformer"]["from_node"] = [0, 1]
    converter.pgm_input_data["transformer"]["to_node"] = [1, 2]
    converter.pgm_output_data["line"]["p_from"] = np.outer(np.array([1.0, 2.0, 4.0], ndmin=2), multiplier)
    converter.pgm_output_data["line"]["q_from"] = np.outer(np.array([0.1, 0.2, 0.4], ndmin=2), multiplier)
    converter.pgm_output_data["line"]["p_to"] = np.outer(np.array([-1.0, -2.0, -4.0], ndmin=2), multiplier)
    converter.pgm_output_data["line"]["q_to"] = np.outer(np.array([-0.1, -0.2, -0.4], ndmin=2), multiplier)
    converter.pgm_output_data["link"]["p_from"] = np.outer(np.array([10.0, 20.0], ndmin=2), multiplier)
    converter.pgm_output_data["link"]["q_from"] = np.outer(np.array([0.01, 0.02], ndmin=2), multiplier)
    converter.pgm_output_data["link"]["p_to"] = np.outer(np.array([-10.0, -20.0], ndmin=2), multiplier)
    converter.pgm_output_data["link"]["q_to"] = np.outer(np.array([-0.01, -0.02], ndmin=2), multiplier)
    converter.pgm_output_data["transformer"]["p_from"] = np.outer(np.array([100.0, 200.0], ndmin=2), multiplier)
    converter.pgm_output_data["transformer"]["q_from"] = np.outer(np.array([0.001, 0.002], ndmin=2), multiplier)
    converter.pgm_output_data["transformer"]["p_to"] = np.outer(np.array([-100.0, -200.0], ndmin=2), multiplier)
    converter.pgm_output_data["transformer"]["q_to"] = np.outer(np.array([-0.001, -0.002], ndmin=2), multiplier)

    p_101 = -1.0 - 10.0 - 100.0
    p_102 = -2.0 - 4.0 + 1.0 - 20.0 + 10.0 - 200.0 + 100.0
    p_103 = 2.0 + 20.0 + 200.0
    p_104 = 4.0
    q_101 = -0.1 - 0.01 - 0.001
    q_102 = -0.2 - 0.4 + 0.1 - 0.02 + 0.01 - 0.002 + 0.001
    q_103 = 0.2 + 0.02 + 0.002
    q_104 = 0.4
    p_arr = np.outer(np.array([p_101, p_102, p_103, p_104], ndmin=2), multiplier)
    q_arr = np.outer(np.array([q_101, q_102, q_103, q_104], ndmin=2), multiplier)
    expected = pd.DataFrame(data=np.hstack((p_arr, q_arr)), columns=power_columns, index=[101, 102, 103, 104]) * 1e-6

    # Act
    converter._pp_buses_output_3ph__accumulate_power(pp_buses_3ph)

    # Assert
    pd.testing.assert_frame_equal(pp_buses_3ph[power_columns], expected)


def test_pp_buses_output_3ph__accumulate_power__output_empty():
    # Arrange
    power_columns = ["p_a_mw", "p_b_mw", "p_c_mw", "q_a_mvar", "q_b_mvar", "q_c_mvar"]
    converter = PandaPowerConverter()
    converter.idx_lookup = {("bus", None): pd.Series([101, 102, 103], index=[0, 1, 2], dtype=np.int32)}
    pp_buses_3ph = pd.DataFrame(np.empty((3, 6), np.float64), columns=power_columns, index=[101, 102, 103])

    converter.pgm_input_data = {
        "link": initialize_array("input", "link", 2),
    }
    converter.pgm_output_data = {
        "link": initialize_array("asym_output", "link", 0),
    }
    converter.pgm_input_data["link"]["from_node"] = [0, 1]
    converter.pgm_input_data["link"]["to_node"] = [1, 2]

    expected = pd.DataFrame(
        np.full((3, 6), fill_value=0.0, dtype=np.float64), columns=power_columns, index=[101, 102, 103]
    )

    # Act
    converter._pp_buses_output_3ph__accumulate_power(pp_buses_3ph)

    # Assert
    pd.testing.assert_frame_equal(pp_buses_3ph[power_columns], expected)
