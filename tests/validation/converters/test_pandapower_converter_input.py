# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import pytest
from power_grid_model import ComponentType, DatasetType
from power_grid_model.data_types import SingleDataset

from power_grid_model_io.converters import PandaPowerConverter
from power_grid_model_io.data_types import ExtraInfo
from power_grid_model_io.utils.json import JsonEncoder

from ...data.pandapower.pp_validation import pp_net, pp_net_3ph_minimal_trafo
from ..utils import compare_extra_info, component_attributes, component_objects, load_json_single_dataset, select_values

VALIDATION_FILE = Path(__file__).parents[2] / "data" / "pandapower" / "pgm_input_data.json"


@lru_cache
def load_and_convert_pp_data() -> Tuple[SingleDataset, ExtraInfo]:
    """
    Load and convert the pandapower validation network
    """
    net = pp_net()
    pp_converter = PandaPowerConverter()
    data, extra_info = pp_converter.load_input_data(net)
    return data, extra_info


@lru_cache
def load_validation_data() -> Tuple[SingleDataset, ExtraInfo]:
    """
    Load the validation data from the json file
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        data, extra_info = load_json_single_dataset(VALIDATION_FILE, data_type="input")

    return data, extra_info


@pytest.fixture
def input_data() -> Tuple[SingleDataset, SingleDataset]:
    """
    Load the pandapower network and the json file, and return the input_data
    """
    actual, _ = load_and_convert_pp_data()
    expected, _ = load_validation_data()
    return actual, expected


@pytest.fixture
def extra_info() -> Tuple[ExtraInfo, ExtraInfo]:
    """
    Load the pandapower network and the json file, and return the extra_info
    """
    _, actual = load_and_convert_pp_data()
    _, expected = load_validation_data()
    return actual, expected


def test_input_data(input_data: Tuple[SingleDataset, SingleDataset]):
    """
    Unit test to preload the expected and actual data
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # Arrange
        actual, expected = input_data

        # Assert
        assert len(expected) <= len(actual)


@pytest.mark.parametrize(("component", "attribute"), component_attributes(VALIDATION_FILE, data_type=DatasetType.input))
def test_attributes(input_data: Tuple[SingleDataset, SingleDataset], component: str, attribute: str):
    """
    For each attribute, check if the actual values are consistent with the expected values
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # Arrange
        actual_data, expected_data = input_data

        # Act
        actual_values, expected_values = select_values(actual_data, expected_data, component, attribute)

        # Assert
        if isinstance(actual_values, pd.Series) and isinstance(expected_values, pd.Series):
            pd.testing.assert_series_equal(actual_values, expected_values)
        else:
            pd.testing.assert_frame_equal(actual_values, expected_values)


@pytest.mark.parametrize(
    ("component", "obj_ids"),
    (pytest.param(component, objects, id=component) for component, objects in component_objects(VALIDATION_FILE)),
)
def test_extra_info(extra_info: Tuple[ExtraInfo, ExtraInfo], component: str, obj_ids: List[int]):
    """
    For each object, check if the actual extra info is consistent with the expected extra info
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # Arrange
        actual, expected = extra_info

        # Assert
        errors = compare_extra_info(actual=actual, expected=expected, component=component, obj_ids=obj_ids)

        # Raise a value error, containing all the errors at once
        if errors:
            raise ValueError("\n" + "\n".join(errors))


def test_extra_info__serializable(extra_info):
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # Arrange
        actual, _expected = extra_info

        # Assert
        json.dumps(actual, cls=JsonEncoder)  # expect no exception


def test_pgm_input_lines__cnf_zero():
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        pp_network = pp_net_3ph_minimal_trafo()
        pp_converter = PandaPowerConverter()
        pp_network.line.c_nf_per_km = 0
        data, _ = pp_converter.load_input_data(pp_network)
        np.testing.assert_array_equal(data[ComponentType.line]["tan1"], 0)

        pp_network.line.c_nf_per_km = 0.001
        pp_network.line.c0_nf_per_km = 0
        data, _ = pp_converter.load_input_data(pp_network)
        np.testing.assert_array_equal(data[ComponentType.line]["tan0"], 0)


@pytest.mark.filterwarnings("error")
def test_simple_example():
    from pandapower.networks import example_simple

    pp_net = example_simple()
    pp_net["gen"] = pp_net["gen"].iloc[:0]
    pp_converter = PandaPowerConverter()
    data, _ = pp_converter.load_input_data(pp_net)
