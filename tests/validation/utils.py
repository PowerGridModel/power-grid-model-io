# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import json
from pathlib import Path
from typing import Generator, Tuple

import numpy as np
import pandas as pd
from power_grid_model.data_types import SingleDataset


def component_attributes(json_path: Path) -> Generator[Tuple[str, str], None, None]:
    """
    Read the json file (only the structure is used, i.e. the component names and attribute name)

    Args:
        json_path: The source json file (e.g. input.json)

    Yields: A tuple (component, attribute) for each attribute available in the json file

    """
    with json_path.open("r") as json_file:
        data = json.load(json_file)

    # Loop over all components in the validation file (in alphabetical order)
    for component, objects in sorted(data.items(), key=lambda x: x[0]):

        # Create a set of attribute names for each object, then take the union of all those sets
        unique_attributes = set().union(*(set(obj.keys()) for obj in objects))

        # Yield the data for each attribute (in alphabetical order)
        for attribute in sorted(unique_attributes):
            yield component, attribute


def select_values(actual: SingleDataset, expected: SingleDataset, component: str, attribute: str):
    """

    Creates two aligned series, for a single component attribute, containinf the actual values and the (non-NaN
    expected values).

    Args:
        actual: Entire dataset containing the actual (i.e. calculated) data
        expected: Entire dataset containing the expected (i.e. validation) data
        component: The component name (e.g. node)
        attribute: The attribute (e.g. u_rated)

    Returns: Two aligned series
    """
    assert component in actual
    assert component in expected
    assert "id" in actual[component].dtype.names
    assert "id" in expected[component].dtype.names
    assert attribute in actual[component].dtype.names
    assert attribute in expected[component].dtype.names

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

    # Return the result
    return actual_values, expected_values
