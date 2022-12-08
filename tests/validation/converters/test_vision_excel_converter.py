# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
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
SOURCE_FILE = DATA_PATH / "vision_{language:s}.xlsx"
VALIDATION_FILE = DATA_PATH / "pgm_input_data_{language:s}.json"
LANGUAGES = ["en", "nl"]
VALIDATION_EN = Path(str(VALIDATION_FILE).format(language="en"))


@lru_cache
def load_and_convert_excel_file(language: str) -> Tuple[SingleDataset, ExtraInfoLookup]:
    """
    Read the excel file and do the conversion
    """
    source_file = Path(str(SOURCE_FILE).format(language=language))
    data, extra_info = VisionExcelConverter(source_file, language=language).load_input_data()
    return data, extra_info


@lru_cache
def load_validation_data(language: str) -> Tuple[SingleDataset, ExtraInfoLookup]:
    """
    Read the excel file and do the conversion
    """
    validation_file = Path(str(VALIDATION_FILE).format(language=language))
    data, extra_info = load_json_single_dataset(validation_file)
    return data, extra_info


@pytest.fixture
def input_data(request) -> Tuple[SingleDataset, SingleDataset]:
    """
    Read the excel file and do the conversion
    """
    actual, _ = load_and_convert_excel_file(language=request.param)
    expected, _ = load_validation_data(language=request.param)
    return actual, expected


@pytest.fixture
def extra_info(request) -> Tuple[ExtraInfoLookup, ExtraInfoLookup]:
    """
    Read the excel file and do the conversion
    """
    _, actual = load_and_convert_excel_file(language=request.param)
    _, expected = load_validation_data(language=request.param)
    return actual, expected


@pytest.mark.parametrize("input_data", LANGUAGES, indirect=True)
def test_input_data(input_data: Tuple[SingleDataset, SingleDataset]):
    """
    Unit test to preload the expected and actual data
    """
    # Arrange
    actual, expected = input_data
    # Assert
    assert len(expected) <= len(actual)


@pytest.mark.parametrize(("component", "attribute"), component_attributes(VALIDATION_EN))
@pytest.mark.parametrize("input_data", LANGUAGES, indirect=True)
def test_attributes(input_data: Tuple[SingleDataset, SingleDataset], component: str, attribute: str):
    """
    For each attribute, check if the actual values are consistent with the expected values
    """
    # Arrange
    actual_data, expected_data = input_data

    # Act
    actual_values, expected_values = select_values(actual_data, expected_data, component, attribute)

    # Assert
    pd.testing.assert_series_equal(actual_values, expected_values)


@pytest.mark.parametrize(
    ("component", "obj_ids"),
    (pytest.param(component, objects, id=component) for component, objects in component_objects(VALIDATION_EN)),
)
@pytest.mark.parametrize("extra_info", LANGUAGES, indirect=True)
def test_extra_info(extra_info: Tuple[ExtraInfoLookup, ExtraInfoLookup], component: str, obj_ids: List[int]):
    """
    For each object, check if the actual extra info is consistent with the expected extra info
    """
    # Arrange
    actual_extra_info, expected_extra_info = extra_info

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


@pytest.mark.parametrize("extra_info", LANGUAGES, indirect=True)
def test_extra_info__serializable(extra_info):
    # Arrange
    actual, _expected = extra_info

    # Assert
    json.dumps(actual, cls=JsonEncoder)  # expect no exception
