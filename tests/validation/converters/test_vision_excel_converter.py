# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from functools import lru_cache
from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest
from power_grid_model.data_types import SingleDataset

from power_grid_model_io.converters.vision_excel_converter import VisionExcelConverter
from power_grid_model_io.data_types import ExtraInfoLookup

from ..utils import component_attributes, load_json_single_dataset, select_values

DATA_PATH = Path(__file__).parents[2] / "data" / "vision"
SOURCE_FILE = DATA_PATH / "vision_validation.xlsx"
VALIDATION_FILE = DATA_PATH / "vision_validation.json"


@pytest.fixture
@lru_cache
def actual() -> SingleDataset:
    """
    Read the excel file and do the conversion (extra info won't be used, for now)
    """

    # Arrange
    actual_data, _actual_extra_info = VisionExcelConverter(SOURCE_FILE).load_input_data()
    return actual_data


@pytest.fixture
def expected() -> SingleDataset:
    """
    Read the json file (extra info is currently not tested and therefore not allowed)
    """

    # Arrange
    expected_data = load_json_single_dataset(VALIDATION_FILE)
    return expected_data


def test_input_data(actual, expected):
    """
    Unit test to preload the expected and actual data
    """
    # Assert
    assert len(expected) <= len(actual)


@pytest.mark.parametrize(("component", "attribute"), component_attributes(DATA_PATH / "vision_validation.json"))
def test_attributes(actual, expected, component: str, attribute: str):
    """
    For each attribute, check if the actual values are consistent with the expected values
    """

    # Act
    actual_values, expected_values = select_values(actual, expected, component, attribute)

    # Assert
    pd.testing.assert_series_equal(actual_values, expected_values)
