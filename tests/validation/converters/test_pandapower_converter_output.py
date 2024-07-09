# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Tuple

import pandapower as pp
import pandas as pd
import pytest
from power_grid_model.validation import assert_valid_input_data

from power_grid_model_io.converters import PandaPowerConverter
from power_grid_model_io.converters.pandapower_converter import PandaPowerData

from ...data.pandapower.pp_validation import pp_net, pp_net_3ph
from ..utils import component_attributes_df, load_json_single_dataset

PGM_OUTPUT_FILE = Path(__file__).parents[2] / "data" / "pandapower" / "pgm_output_data.json"
PGM_ASYM_OUTPUT_FILE = Path(__file__).parents[2] / "data" / "pandapower" / "pgm_asym_output_data.json"


@contextmanager
def temporary_file_cleanup(file_path):
    try:
        yield
    finally:
        os.remove(file_path)


@lru_cache
def load_and_convert_pgm_data() -> PandaPowerData:
    """
    Load and convert the power_grid_model results
    """
    data, extra_info = load_json_single_dataset(PGM_OUTPUT_FILE, data_type="sym_output")
    converter = PandaPowerConverter()
    return converter.convert(data=data, extra_info=extra_info)


@lru_cache
def load_and_convert_pgm_data_3ph() -> PandaPowerData:
    """
    Load and convert the power_grid_model results
    """
    data, extra_info = load_json_single_dataset(PGM_ASYM_OUTPUT_FILE, data_type="asym_output")
    converter = PandaPowerConverter()
    return converter.convert(data=data, extra_info=extra_info)


@lru_cache
def load_validation_data() -> PandaPowerData:
    """
    Load the validation data from the pp file
    """
    net = pp_net()
    pp.runpp(net, calculate_voltage_angles=True, tolerance_mva=1e-10, trafo_model="pi", trafo_loading="power")
    return net


@lru_cache
def load_validation_data_3ph() -> PandaPowerData:
    """
    Load the validation data from the pp file
    """
    net = pp_net_3ph()
    pp.runpp_3ph(net, calculate_voltage_angles=True, tolerance_mva=1e-10, trafo_loading="power")
    return net


@pytest.fixture
def output_data() -> Tuple[PandaPowerData, PandaPowerData]:
    """
    Load the pandapower network and the json file, and return the output_data
    """
    actual = load_and_convert_pgm_data()
    expected = load_validation_data()
    return actual, expected


@pytest.fixture
def output_data_3ph() -> Tuple[PandaPowerData, PandaPowerData]:
    """
    Load the pandapower network and the json file, and return the output_data
    """
    actual = load_and_convert_pgm_data_3ph()
    expected = load_validation_data_3ph()
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
