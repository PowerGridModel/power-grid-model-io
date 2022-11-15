# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import json
from pathlib import Path
from typing import Generator

import numpy as np
import pandas as pd
import pytest
from power_grid_model.utils import import_json_data

from power_grid_model_io.converters.vision_excel_converter import VisionExcelConverter

DATA_PATH = Path(__file__).parents[2] / "data" / "vision"


def available_attributes() -> Generator[pytest.param, None, None]:
    # Read the excel file and do the conversion (extra info won't be used)
    actual, _extra_info = VisionExcelConverter(DATA_PATH / "vision_validation.xlsx").load_input_data()

    # Read the json file (extra info is currently not tested and therefore not allowed)
    try:
        expected = import_json_data(DATA_PATH / "vision_validation.json", data_type="input", ignore_extra=False)
    except ValueError as ex:
        if "ignore_extra=True" in str(ex):
            raise ValueError("Extra info is currently not tested and therefore not allowed") from ex
        else:
            raise

    # Read the json file again (only the structure is used, i.e. the component names and attribute name)
    with (DATA_PATH / "vision_validation.json").open("r") as json_file:
        structure = json.load(json_file)

    # Loop over all components in the validation file (in alphabetical order)
    for component, objects in sorted(structure.items(), key=lambda x: x[0]):

        # List the attributes
        attributes = set().union(*(set(obj.keys()) for obj in objects))

        # Loop over all attributes that exist both in the validation file
        for attribute in sorted(attributes):
            # Create an index series for both the actual data and the expected data
            actual_values = pd.Series(actual[component][attribute], index=actual[component]["id"]).sort_index()
            expected_values = pd.Series(expected[component][attribute], index=expected[component]["id"]).sort_index()

            # Create a selection mask, to select only non-NaN values in the validation file
            if expected_values.dtype == np.int8:
                mask = expected_values != -(2**7)
            elif expected_values.dtype == np.int32:
                mask = expected_values != -(2**31)
            elif expected_values.dtype == np.int64:
                mask = expected_values != -(2**63)
            else:
                mask = ~pd.isna(expected_values)

            # Use only the actual_values for which we have expected values
            actual_values = actual_values[expected_values.index][mask]
            expected_values = expected_values[mask]

            # Yield the result as a param instance
            yield pytest.param(actual_values, expected_values, id=f"{component}-{attribute}")


@pytest.mark.parametrize(("actual", "expected"), available_attributes())
def test_input_data(actual: pd.Series, expected: pd.Series):
    pd.testing.assert_series_equal(actual, expected)
