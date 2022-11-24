# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import json
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pytest
from power_grid_model.data_types import SingleDataset

from power_grid_model_io.converters.vision_excel_converter import VisionExcelConverter
from power_grid_model_io.data_types import ExtraInfoLookup
from power_grid_model_io.utils.json import JsonEncoder

from ..utils import component_attributes, component_objects, load_json_single_dataset, select_values

DATA_PATH = Path(__file__).parents[2] / "data" / "vision"
SOURCE_FILE = DATA_PATH / "vision_validation.xlsx"
VALIDATION_FILE = DATA_PATH / "vision_validation.json"


@pytest.fixture
@lru_cache
def actual() -> Tuple[SingleDataset, ExtraInfoLookup]:
    """
    Read the excel file and do the conversion (extra info won't be used, for now)
    """
    actual_data, actual_extra_info = VisionExcelConverter(SOURCE_FILE).load_input_data()
    return actual_data, actual_extra_info


@pytest.fixture
def expected() -> Tuple[SingleDataset, ExtraInfoLookup]:
    """
    Read the json file (extra info is currently not tested and therefore not allowed)
    """
    expected_data, expected_extra_info = load_json_single_dataset(VALIDATION_FILE)
    return expected_data, expected_extra_info


def test_input_data(actual, expected):
    """
    Unit test to preload the expected and actual data
    """
    # Assert
    assert len(expected) <= len(actual)


@pytest.mark.parametrize(("component", "attribute"), component_attributes(VALIDATION_FILE))
def test_attributes(actual, expected, component: str, attribute: str):
    """
    For each attribute, check if the actual values are consistent with the expected values
    """
    # Arrange
    actual_data, _ = actual
    expected_data, _ = expected

    # Act
    actual_values, expected_values = select_values(actual_data, expected_data, component, attribute)

    # Assert
    pd.testing.assert_series_equal(actual_values, expected_values)


@pytest.mark.parametrize(
    ("component", "obj_ids"),
    (pytest.param(component, objects, id=component) for component, objects in component_objects(VALIDATION_FILE)),
)
def test_extra_info(actual, expected, component: str, obj_ids: List[int]):
    """
    For each object, check if the actual extra info is consistent with the expected extra info
    """
    # Arrange
    _, actual_extra_info = actual
    _, expected_extra_info = expected

    # Assert

    # We'll collect all errors, instead of terminating at the first error
    errors = []

    # Check each object in this component
    for obj_id in obj_ids:

        # If there is no extra_info available in the validation data, just skip this object
        if obj_id not in expected_extra_info:
            continue

        # If the object doesn't exist in the actual data, that's an error
        if obj_id not in actual_extra_info:
            errors.append(f"Expected {component} #{obj_id}, but it is missing.")
            continue

        # Now for each extra_info in the validation file, check if it matches the actual extra info
        act = actual_extra_info[obj_id]
        for key, value in expected_extra_info[obj_id].items():

            # If the extra_info doesn't exist, that's an error
            if key not in act:
                errors.append(f"Expected extra info '{key}' for {component} #{obj_id}, but it is missing.")

            # If the values don't match, that's an error
            elif act[key] != value:
                errors.append(
                    f"Expected extra info '{key}' for {component} #{obj_id} to be {value}, " f"but it is {act[key]}."
                )

    # Raise a value error, containing all the errors at once
    if errors:
        raise ValueError("\n" + "\n".join(errors))


def test_extra_info__serializable(actual):
    # Arrange
    _, extra_info = actual

    # Assert
    json.dumps(extra_info, cls=JsonEncoder)  # expect no exception
