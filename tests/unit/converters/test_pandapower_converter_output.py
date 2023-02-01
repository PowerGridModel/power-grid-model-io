# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Callable
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from power_grid_model import initialize_array

from power_grid_model_io.converters.pandapower_converter import PandaPowerConverter


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
    assert len(converter.method_calls) == 8
    converter._pp_buses_output.assert_called_once_with()
    converter._pp_lines_output.assert_called_once_with()
    converter._pp_ext_grids_output.assert_called_once_with()
    converter._pp_loads_output.assert_called_once_with()
    converter._pp_shunts_output.assert_called_once_with()
    converter._pp_trafos_output.assert_called_once_with()
    converter._pp_sgens_output.assert_called_once_with()
    converter._pp_trafos3w_output.assert_called_once_with()


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
    pgm_output_attributes = ["id", "u_pu", "u_angle", ""]  # Left blank because this part depends on what kind of
    # Branch the node is connected to. However, If the node_injection becomes finished then it will be easier to input
    # a specific attribute
    pp_output_attributes = ["vm_pu", "va_degree", "p_mw", "q_mvar"]

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
    # for pgm attributes I did not include any attributes that are taken from nodes in node_lookup
    # However I included some attributes that were taken from pgm_input_lines such as: from_node, to_node
    pgm_output_attributes = [
        "id",
        "from_node",
        "to_node",
        "p_from",
        "q_from",
        "p_to",
        "q_to",
        "i_from",
        "i_to",
        "loading",
    ]
    pp_output_attributes = [
        "p_from_mw",
        "q_from_mvar",
        "p_to_mw",
        "q_to_mvar",
        "pl_mw",
        "ql_mvar",
        "i_from_ka",
        "i_to_ka",
        "i_ka",
        "vm_from_pu",
        "vm_to_pu",
        "va_from_degree",
        "va_to_degree",
        "loading_percent",
    ]

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
    pgm_output_attributes = ["id", "p", "q"]
    pp_output_attributes = ["p_mw", "q_mvar"]

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
    pgm_output_attributes = ["id", "node", "p", "q"]  # node is taken from pgm_input_shunts
    pp_output_attributes = ["p_mw", "q_mvar", "vm_pu"]

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
    pgm_output_attributes = ["id", "p", "q"]
    pp_output_attributes = ["p_mw", "q_mvar"]

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


def test_output_trafos(converter):
    pgm_output_attributes = [  # from node and to node are taken from pgm_input_transformers
        "id",
        "from_node",
        "to_node",
        "p_from",
        "q_from",
        "p_to",
        "q_to",
        "i_from",
        "i_to",
        "loading",
    ]
    pp_output_attributes = [
        "p_hv_mw",
        "q_hv_mvar",
        "p_lv_mw",
        "q_lv_mvar",
        "pl_mw",
        "ql_mvar",
        "i_hv_ka",
        "i_lv_ka",
        "vm_hv_pu",
        "vm_lv_pu",
        "va_hv_degree",
        "va_lv_degree",
        "loading_percent",
    ]

    # Arrange
    mock_pgm_array = MagicMock()
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
        # mock_pgm_array.__getitem__.assert_any_call("u_pu")
        # mock_pgm_array.__getitem__.assert_any_call("u_degree")
        mock_pgm_array.__getitem__.assert_any_call("loading")

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


def test_output_trafo3w(converter):
    pgm_output_attributes = [  # "node_1", "node_2", "node_3" are taken from pgm_input_transformers3w
        "id",
        "node_1",
        "node_2",
        "node_3",
        "p_1",
        "q_1",
        "p_2",
        "q_2",
        "p_3",
        "q_3",
        "i_1",
        "i_2",
        "i_3",
        "loading",
    ]
    pp_output_attributes = [
        "p_hv_mw",
        "q_hv_mvar",
        "p_mv_mw",
        "q_mv_mvar",
        "p_lv_mw",
        "q_lv_mvar",
        "pl_mw",
        "ql_mvar",
        "i_hv_ka",
        "i_mv_ka",
        "i_lv_ka",
        "vm_hv_pu",
        "vm_mv_pu",
        "vm_lv_pu",
        "va_hv_degree",
        "va_mv_degree",
        "va_lv_degree",
        "loading_percent",
    ]

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


def test_output_load():
    pgm_output_attributes = ["id", "p", "q"]
    pp_output_attributes = ["p_mw", "q_mvar"]

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


def test_output_asymmetric_load(converter):
    pgm_output_attributes = ["id", "p", "q"]
    pp_output_attributes = ["p_a_mw", "q_a_mvar", "p_b_mw", "q_b_mvar", "p_c_mw", "q_c_mvar"]

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
        mock_pp_df.return_value.__setitem__.assert_any_call("p_a_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_a_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_b_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_b_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_c_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_c_mvar", ANY)

        # result
        converter.pp_output_data.__setitem__.assert_called_once_with("res_asymmetric_load", mock_pp_df.return_value)


def test_output_asymmetric_sgen(converter):
    pgm_output_attributes = ["id", "p", "q"]
    pp_output_attributes = ["p_a_mw", "q_a_mvar", "p_b_mw", "q_b_mvar", "p_c_mw", "q_c_mvar"]

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
        mock_pp_df.return_value.__setitem__.assert_any_call("p_a_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_a_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_b_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_b_mvar", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("p_c_mw", ANY)
        mock_pp_df.return_value.__setitem__.assert_any_call("q_c_mvar", ANY)

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
