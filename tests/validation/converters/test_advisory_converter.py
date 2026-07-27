# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from functools import lru_cache
from pathlib import Path

import pandas as pd
import pytest
from power_grid_model import AttributeType as AT, ComponentType as CT, DatasetType
from power_grid_model.data_types import SingleDataset

from power_grid_model_io.converters import VisionExcelConverter
from power_grid_model_io.data_types import ExtraInfo
from tests.validation.utils import (
    compare_extra_info,
    component_attributes,
    component_objects,
    load_json_single_dataset,
    select_values,
)

DATA_PATH = Path(__file__).parents[2] / "data" / "vision"
SOURCE_FILE = DATA_PATH / "vision_advisory.xlsx"
VALIDATION_FILE = DATA_PATH / "pgm_input_data_advisory.json"
HACKS_AND_MAPPINGS = {
    "cel": DATA_PATH / "vision_advisory.yaml",
    "allowlist": DATA_PATH / "vision_advisory_allowlist.yaml",
}


@lru_cache
def vision_excel_converter(hack: str) -> VisionExcelConverter:
    """
    Read the excel and mapping files. Create and return converter.
    """
    return VisionExcelConverter(source_file=SOURCE_FILE, mapping_file=HACKS_AND_MAPPINGS[hack], hack=hack)


@lru_cache
def load_and_convert_excel_file(hack: str) -> tuple[SingleDataset, ExtraInfo]:
    """
    Convert the excel file. Return converted data and additional info.
    """
    data, extra_info = vision_excel_converter(hack).load_input_data()
    return data, extra_info


@lru_cache
def load_validation_data() -> tuple[SingleDataset, ExtraInfo]:
    """
    Load the validation data from the json file and convert to PGM format. Return converted data and additional info.
    """
    data, extra_info = load_json_single_dataset(file_path=VALIDATION_FILE, data_type=DatasetType.input)
    return data, extra_info


@pytest.fixture(params=[hack for hack in HACKS_AND_MAPPINGS])
def hack(request) -> str:
    return request.param


@pytest.fixture
def input_data(hack: str) -> tuple[SingleDataset, SingleDataset]:
    """
    Get converted data from the excel file and the validation data from the json file.
    """
    actual, _ = load_and_convert_excel_file(hack)
    expected, _ = load_validation_data()
    return actual, expected


@pytest.fixture
def extra_info(hack: str) -> tuple[ExtraInfo, ExtraInfo]:
    """
    Get converted extra info from the excel file and the validation extra info from the json file.
    """
    _, actual = load_and_convert_excel_file(hack)
    _, expected = load_validation_data()
    return actual, expected


@pytest.mark.parametrize(
    ("component", "attribute"), list(component_attributes(json_path=VALIDATION_FILE, data_type=DatasetType.input))
)
def test_attributes(input_data: tuple[SingleDataset, SingleDataset], component: CT, attribute: AT):
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
    [
        pytest.param(component, objects, id=component)
        for component, objects in component_objects(json_path=VALIDATION_FILE)
    ],
)
def test_extra_info(extra_info: tuple[ExtraInfo, ExtraInfo], component: CT, obj_ids: list[int]):
    """
    For each object, check if the actual extra info is consistent with the expected extra info
    """
    # Arrange
    actual, expected = extra_info

    # Assert
    errors = compare_extra_info(actual=actual, expected=expected, component=component, obj_ids=obj_ids)

    # Raise a value error, containing all the errors at once
    if errors:
        raise ValueError("\n" + "\n".join(errors))
