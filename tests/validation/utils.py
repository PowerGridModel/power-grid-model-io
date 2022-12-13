# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import json
from functools import lru_cache
from pathlib import Path
from typing import Generator, List, Tuple

import numpy as np
import pandas as pd
from power_grid_model import power_grid_meta_data
from power_grid_model.data_types import SingleDataset, SinglePythonDataset
from power_grid_model.utils import convert_python_single_dataset_to_single_dataset

from power_grid_model_io.data_types import ExtraInfoLookup, StructuredData


@lru_cache()
def load_json_file(file_path: Path) -> StructuredData:
    """
    Load (and cache) a json file
    Args:
        file_path: The path to the json file

    Returns: The parsed contents of the json file in a native python structure
    """
    with file_path.open(mode="r", encoding="utf-8") as json_file:
        data = json.load(json_file)
    return data


def component_objects(json_path: Path) -> Generator[Tuple[str, List[int]], None, None]:
    """
    Read the json file (only the components and their ids are are used, i.e. the component names and attribute name)

    Args:
        json_path: The source json file (e.g. input.json)

    Yields: A tuple (component, ids) for each component available in the json file

    """
    data = load_json_file(json_path)
    assert isinstance(data, dict)

    # Loop over all components in the validation file (in alphabetical order)
    for component, objects in sorted(data.items(), key=lambda x: x[0]):
        obj_ids = [obj["id"] for obj in objects]
        if obj_ids:
            yield component, obj_ids


def component_attributes(json_path: Path, data_type: str = "input") -> Generator[Tuple[str, str], None, None]:
    """
    Read the json file (only the structure is used, i.e. the component names and attribute name)

    Args:
        json_path: The source json file (e.g. input.json)
        data_type: The pgm data type; this determines which attributes are considered pgm attributes

    Yields: A tuple (component, attribute) for each attribute available in the json file

    """
    data = load_json_file(json_path)
    assert isinstance(data, dict)

    # Loop over all components in the validation file (in alphabetical order)
    for component, objects in sorted(data.items(), key=lambda x: x[0]):

        # Create a set of attribute names for each object, then take the union of all those sets
        pgm_attr = set(power_grid_meta_data[data_type][component]["dtype"].names)
        obj_keys = (set(obj.keys()) & pgm_attr for obj in objects)
        unique_attributes = set().union(*obj_keys)

        # Yield the data for each attribute (in alphabetical order)
        for attribute in sorted(unique_attributes):
            yield component, attribute


def select_values(actual: SingleDataset, expected: SingleDataset, component: str, attribute: str):
    """

    Creates two aligned series, for a single component attribute, containing the actual values and the (non-NaN
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
    if np.issubdtype(expected_values.dtype, np.integer):
        mask = expected_values != np.iinfo(expected_values.dtype).min
    else:
        mask = ~pd.isna(expected_values)

    # Use only the actual_values for which we have expected values
    missing_idx = set(expected_values.index) - set(actual_values.index)
    if len(missing_idx) == 1:
        raise KeyError(f"Expected {component} #{missing_idx.pop()}, but it is missing.")
    elif len(missing_idx) > 1:
        raise KeyError(f"Expected {component}s {missing_idx}, but they are missing.")

    actual_values = actual_values[expected_values.index][mask]
    expected_values = expected_values[mask]

    # Return the result
    return actual_values, expected_values


def extract_extra_info(data: SinglePythonDataset, data_type: str = "input") -> ExtraInfoLookup:
    """
    Reads the dataset and collect all arguments that aren't pgm attributes

    Args:
        data: A dictionary containing all the components
        data_type: The pgm data type; this determines which attributes are considered pgm attributes

    Returns: A dictionary indexed on the object id and containing the extra info for all the objects.
    """
    extra_info: ExtraInfoLookup = {}
    for component, objects in data.items():
        pgm_attr = set(power_grid_meta_data[data_type][component]["dtype"].names)
        for obj in objects:
            obj_extra_info = {attr: val for attr, val in obj.items() if attr not in pgm_attr}
            if obj_extra_info:
                extra_info[obj["id"]] = obj_extra_info
    return extra_info


def load_json_single_dataset(file_path: Path, data_type: str = "input") -> Tuple[SingleDataset, ExtraInfoLookup]:
    """
    Loads and parses a json file in the most basic way, without using power_grid_model_io functions.

    Args:
        file_path: The json file path
        data_type: The pgm data type; this determines which attributes are considered pgm attributes

    Returns: A native pgm dataset and an extra info lookup table

    """
    raw_data = load_json_file(file_path)
    assert isinstance(raw_data, dict)
    dataset = convert_python_single_dataset_to_single_dataset(data=raw_data, data_type=data_type, ignore_extra=True)
    extra_info = extract_extra_info(raw_data)
    return dataset, extra_info
