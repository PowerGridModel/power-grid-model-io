# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest
from power_grid_model.data_types import SingleDataset
from power_grid_model.utils import import_json_data

from power_grid_model_io.converters.vision_excel_converter import VisionExcelConverter

from ..utils import component_attributes, select_values

DATA_PATH = Path(__file__).parents[2] / "data" / "vision"


@lru_cache
def load_actual() -> SingleDataset:
    """
    Read the excel file and do the conversion (extra info won't be used, for now)
    """
    actual_data, _extra_info = VisionExcelConverter(DATA_PATH / "vision_validation.xlsx").load_input_data()
    return actual_data


@lru_cache
def load_expected() -> SingleDataset:
    """
    Read the json file (extra info is currently not tested and therefore not allowed)
    """
    try:
        expected_data = import_json_data(DATA_PATH / "vision_validation.json", data_type="input", ignore_extra=False)
    except ValueError as ex:
        if "ignore_extra=True" in str(ex):
            raise ValueError("Extra info is currently not tested and therefore not allowed") from ex
        else:
            raise
    return expected_data


def test_input_data():
    """
    Unit test to preload the expected and actual data
    """
    # Arrange
    expected = load_expected()

    # Act
    actual = load_actual()

    # Assert
    assert len(expected) <= len(actual)


@pytest.mark.parametrize(("component", "attribute"), component_attributes(DATA_PATH / "vision_validation.json"))
def test_attributes(component: str, attribute: str):
    """
    For each attribute, check if the actual values are consistent with the expected values
    """
    # Arrange / Act
    actual, expected = select_values(load_actual(), load_expected(), component, attribute)

    # Assert
    pd.testing.assert_series_equal(actual, expected)
