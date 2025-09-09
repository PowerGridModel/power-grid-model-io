# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest
from power_grid_model.validation import assert_valid_input_data

from power_grid_model_io.converters import PandaPowerConverter
from power_grid_model_io.converters.pandapower_converter import PandaPowerData

from ...data.pandapower.pp_validation import pp_net, pp_net_3ph, pp_net_3ph_minimal_trafo
from ..utils import component_attributes_df, load_json_single_dataset

pp = pytest.importorskip("pandapower", reason="pandapower is not installed")
# we add this to enable python 3.13 testing even though pandapower 3.0 is not yet compatible with it

PGM_PP_TEST_DATA = Path(__file__).parents[2] / "data" / "pandapower"
PGM_OUTPUT_FILE = PGM_PP_TEST_DATA / "pgm_output_data.json"
PGM_ASYM_OUTPUT_FILE = PGM_PP_TEST_DATA / "pgm_asym_output_data.json"
PP_V2_NET_OUTPUT_FILE = PGM_PP_TEST_DATA / "pp_v2_net_output.json"
PP_V2_NET_3PH_OUTPUT_FILE = PGM_PP_TEST_DATA / "pp_v2_net_3ph_output.json"
PP_V2_NET_3PH_OUTPUT_FILE_CURRENT_LOADING = PGM_PP_TEST_DATA / "pp_v2_net_3ph_output_current_loading.json"


@contextmanager
def temporary_file_cleanup(file_path):
    try:
        yield
    finally:
        os.remove(file_path)


@lru_cache
def load_and_convert_pgm_data(trafo_loading="power") -> PandaPowerData:
    """
    Load and convert the power_grid_model results
    """
    data, _ = load_json_single_dataset(PGM_OUTPUT_FILE, data_type="sym_output")
    converter = PandaPowerConverter(trafo_loading=trafo_loading)
    converter.load_input_data(load_validation_data(), make_extra_info=False)
    return converter.convert(data=data)


@lru_cache
def load_and_convert_pgm_data_3ph(trafo_loading="power") -> PandaPowerData:
    """
    Load and convert the power_grid_model results
    """
    data, _ = load_json_single_dataset(PGM_ASYM_OUTPUT_FILE, data_type="asym_output")
    converter = PandaPowerConverter(trafo_loading=trafo_loading)
    converter.load_input_data(load_validation_data_3ph(), make_extra_info=False)
    return converter.convert(data=data)


@lru_cache
def load_validation_data() -> PandaPowerData:
    """
    Load the validation data from the pp file
    """
    return pp.file_io.from_json(PP_V2_NET_OUTPUT_FILE)


@lru_cache
def load_validation_data_3ph(trafo_loading="power") -> PandaPowerData:
    """
    Load the validation data from the pp file
    """
    if trafo_loading == "power":
        return pp.file_io.from_json(PP_V2_NET_3PH_OUTPUT_FILE)
    else:
        return pp.file_io.from_json(PP_V2_NET_3PH_OUTPUT_FILE_CURRENT_LOADING)


@pytest.fixture(params=["power", "current"])
def output_data(request) -> Tuple[PandaPowerData, PandaPowerData]:
    """
    Load the pandapower network and the json file, and return the output_data
    """
    actual = load_and_convert_pgm_data(request.param)
    expected = load_validation_data()
    return actual, expected


@pytest.fixture(params=["power", "current"])
def output_data_3ph(request) -> Tuple[PandaPowerData, PandaPowerData]:
    """
    Load the pandapower network and the json file, and return the output_data
    """
    actual = load_and_convert_pgm_data_3ph(request.param)
    expected = load_validation_data_3ph(request.param)
    return actual, expected


def test_generate_output():  # TODO: REMOVE THIS FUNCTION
    from power_grid_model import PowerGridModel

    from power_grid_model_io.converters import PgmJsonConverter

    net = pp_net()
    converter = PandaPowerConverter()
    input_data, extra_info = converter.load_input_data(net)
    assert_valid_input_data(input_data=input_data)
    pgm = PowerGridModel(input_data=input_data)
    output_data = pgm.calculate_power_flow()
    temp_file = PGM_OUTPUT_FILE.with_name(PGM_OUTPUT_FILE.stem + "_temp").with_suffix(".json")
    with temporary_file_cleanup(temp_file):
        json_converter = PgmJsonConverter(destination_file=temp_file)
        json_converter.save(data=output_data, extra_info=extra_info)


def test_generate_output_3ph():  # TODO: REMOVE THIS FUNCTION
    from power_grid_model import PowerGridModel

    from power_grid_model_io.converters import PgmJsonConverter

    net = pp_net_3ph()
    converter = PandaPowerConverter()
    input_data, extra_info = converter.load_input_data(net)
    assert_valid_input_data(input_data=input_data, symmetric=False)
    pgm = PowerGridModel(input_data=input_data)
    output_data_asym = pgm.calculate_power_flow(symmetric=False)
    temp_file = PGM_ASYM_OUTPUT_FILE.with_name(PGM_ASYM_OUTPUT_FILE.stem + "_temp").with_suffix(".json")
    with temporary_file_cleanup(temp_file):
        json_converter = PgmJsonConverter(destination_file=temp_file)
        json_converter.save(data=output_data_asym, extra_info=extra_info)


def test_output_trafos_3ph__power__with_comparison():
    import numpy as np

    def check_result(net):
        v_hv = net.trafo.vn_hv_kv
        v_lv = net.trafo.vn_lv_kv
        i_max_hv = np.divide(net.trafo.sn_mva, v_hv * np.sqrt(3)) * 1e3
        i_max_lv = np.divide(net.trafo.sn_mva, v_lv * np.sqrt(3)) * 1e3

        i_a_hv = net.res_trafo_3ph.loc[:, "i_a_hv_ka"] * 1000
        i_b_hv = net.res_trafo_3ph.loc[:, "i_b_hv_ka"] * 1000
        i_c_hv = net.res_trafo_3ph.loc[:, "i_c_hv_ka"] * 1000

        i_a_lv = net.res_trafo_3ph.loc[:, "i_a_lv_ka"] * 1000
        i_b_lv = net.res_trafo_3ph.loc[:, "i_b_lv_ka"] * 1000
        i_c_lv = net.res_trafo_3ph.loc[:, "i_c_lv_ka"] * 1000

        np.testing.assert_allclose(
            np.maximum(i_a_hv / i_max_hv, i_a_lv / i_max_lv) * 100, net.res_trafo_3ph.loading_a_percent
        )
        np.testing.assert_allclose(
            np.maximum(i_b_hv / i_max_hv, i_b_lv / i_max_lv) * 100, net.res_trafo_3ph.loading_b_percent
        )
        np.testing.assert_allclose(
            np.maximum(i_c_hv / i_max_hv, i_c_lv / i_max_lv) * 100, net.res_trafo_3ph.loading_c_percent
        )
        np.testing.assert_allclose(
            np.maximum(
                np.maximum(net.res_trafo_3ph.loading_a_percent, net.res_trafo_3ph.loading_b_percent),
                net.res_trafo_3ph.loading_c_percent,
            ),
            net.res_trafo_3ph.loading_percent,
        )
        np.testing.assert_allclose(
            np.maximum(
                np.maximum(net.res_line_3ph.loading_a_percent, net.res_line_3ph.loading_b_percent),
                net.res_line_3ph.loading_c_percent,
            ),
            net.res_line_3ph.loading_percent,
        )

    def compare_result(actual, expected, *, rtol):
        np.testing.assert_allclose(actual.trafo.vn_hv_kv, expected.trafo.vn_hv_kv, rtol=rtol)
        np.testing.assert_allclose(actual.trafo.vn_lv_kv, expected.trafo.vn_lv_kv, rtol=rtol)
        np.testing.assert_allclose(actual.trafo.sn_mva, expected.trafo.sn_mva, rtol=rtol)
        np.testing.assert_allclose(
            actual.res_trafo_3ph.loc[:, "i_a_hv_ka"], expected.res_trafo_3ph.loc[:, "i_a_hv_ka"], rtol=rtol
        )
        np.testing.assert_allclose(
            actual.res_trafo_3ph.loc[:, "i_b_hv_ka"], expected.res_trafo_3ph.loc[:, "i_b_hv_ka"], rtol=rtol
        )
        np.testing.assert_allclose(
            actual.res_trafo_3ph.loc[:, "i_c_hv_ka"], expected.res_trafo_3ph.loc[:, "i_c_hv_ka"], rtol=rtol
        )
        np.testing.assert_allclose(
            actual.res_trafo_3ph.loc[:, "i_a_lv_ka"], expected.res_trafo_3ph.loc[:, "i_a_lv_ka"], rtol=rtol
        )
        np.testing.assert_allclose(
            actual.res_trafo_3ph.loc[:, "i_b_lv_ka"], expected.res_trafo_3ph.loc[:, "i_b_lv_ka"], rtol=rtol
        )
        np.testing.assert_allclose(
            actual.res_trafo_3ph.loc[:, "i_c_lv_ka"], expected.res_trafo_3ph.loc[:, "i_c_lv_ka"], rtol=rtol
        )
        np.testing.assert_allclose(
            actual.res_trafo_3ph.loading_a_percent, expected.res_trafo_3ph.loading_a_percent, rtol=rtol
        )
        np.testing.assert_allclose(
            actual.res_trafo_3ph.loading_b_percent, expected.res_trafo_3ph.loading_b_percent, rtol=rtol
        )
        np.testing.assert_allclose(
            actual.res_trafo_3ph.loading_c_percent, expected.res_trafo_3ph.loading_c_percent, rtol=rtol
        )

    pgm_net = pp_net_3ph_minimal_trafo()
    pp_net = pp_net_3ph_minimal_trafo()
    # Asymmetric Load
    pp.runpp_pgm(pgm_net, symmetric=False)
    pp.runpp_3ph(pp_net)
    check_result(pgm_net)
    check_result(pp_net)
    compare_result(pgm_net, pp_net, rtol=0.04)

    # Symmetric Load
    pgm_net.asymmetric_load.loc[:, ["p_a_mw", "p_b_mw", "p_c_mw"]] = 0.2
    pgm_net.asymmetric_load.loc[:, ["q_a_mvar", "q_b_mvar", "q_c_mar"]] = 0.05
    pp_net.asymmetric_load.loc[:, ["p_a_mw", "p_b_mw", "p_c_mw"]] = 0.2
    pp_net.asymmetric_load.loc[:, ["q_a_mvar", "q_b_mvar", "q_c_mar"]] = 0.05
    pp.runpp_pgm(pgm_net, symmetric=False)
    pp.runpp_3ph(pp_net)
    check_result(pgm_net)
    check_result(pp_net)
    compare_result(pgm_net, pp_net, rtol=0.005)


def test_output_data(output_data: Tuple[PandaPowerData, PandaPowerData]):
    """
    Unit test to preload the expected and actual data
    """
    # Arrange
    actual, expected = output_data

    # Assert
    assert all(key in expected for key in actual)


def test_output_data_3ph(output_data_3ph: Tuple[PandaPowerData, PandaPowerData]):
    """
    Unit test to preload the expected and actual data for asym output
    """
    # Arrange
    actual, expected = output_data_3ph

    # Assert
    assert all(key in expected for key in actual)


@pytest.mark.parametrize(("component", "attribute"), component_attributes_df(load_and_convert_pgm_data()))
def test_attributes(output_data: Tuple[PandaPowerData, PandaPowerData], component: str, attribute: str):
    """
    For each attribute, check if the actual values are consistent with the expected values
    """
    # Arrange
    actual_data, expected_data = output_data

    # Act
    actual_values = actual_data[component][attribute]
    expected_values = expected_data[component][attribute]

    # Assert
    pd.testing.assert_series_equal(actual_values, expected_values, atol=5e-4, rtol=1e-4)


@pytest.mark.parametrize(("component", "attribute"), component_attributes_df(load_and_convert_pgm_data_3ph()))
def test_attributes_3ph(output_data_3ph: Tuple[PandaPowerData, PandaPowerData], component: str, attribute: str):
    """
    For each attribute, check if the actual values are consistent with the expected values for asym
    """
    # Arrange
    actual_data, expected_data = output_data_3ph

    # Act
    actual_values = actual_data[component][attribute]
    expected_values = expected_data[component][attribute]

    # Assert
    pd.testing.assert_series_equal(actual_values, expected_values, atol=5e-4, rtol=1e-4)
