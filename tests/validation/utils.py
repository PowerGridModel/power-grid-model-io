# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import json
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Generator, List, Mapping, Tuple

import numpy as np
import pandas as pd
from power_grid_model import power_grid_meta_data
from power_grid_model.data_types import SingleDataset, SinglePythonDataset
from power_grid_model.errors import PowerGridSerializationError
from power_grid_model.utils import import_json_data, json_deserialize_from_file

from power_grid_model_io.data_types import ExtraInfo, StructuredData


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


def component_attributes(json_path: Path, data_type: str) -> Generator[Tuple[str, str], None, None]:
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
        pgm_attr = set(power_grid_meta_data[data_type][component].dtype.names)
        obj_keys = (set(obj.keys()) & pgm_attr for obj in objects)
        unique_attributes = set().union(*obj_keys)

        # Yield the data for each attribute (in alphabetical order)
        for attribute in sorted(unique_attributes):
            yield component, attribute


def component_attributes_df(data: Mapping[str, pd.DataFrame]) -> Generator[Tuple[str, str], None, None]:
    """
    Extract the component and attribute names from the DataFrames

    Args:
        data: A dictionary of pandas DataFrames

    Yields: A tuple (component, attribute) for each attribute available in the json file

    """

    for component, df in sorted(data.items(), key=lambda x: x[0]):
        for attribute in sorted(df.columns):
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
    actual_attr = actual[component][attribute]
    expected_attr = expected[component][attribute]
    assert actual_attr.ndim == expected_attr.ndim in [1, 2]
    pd_data_fn = pd.DataFrame if actual_attr.ndim == 2 else pd.Series
    actual_values = pd_data_fn(actual_attr, index=actual[component]["id"]).sort_index()
    expected_values = pd_data_fn(expected_attr, index=expected[component]["id"]).sort_index()

    # Create a selection mask, to select only non-NaN values in the validation file
    if np.issubdtype(expected_values.values.dtype, np.integer):
        mask = expected_values != np.iinfo(expected_values.values.dtype).min
    else:
        mask = ~pd.isna(expected_values)

    # Use only the actual_values for which we have expected values
    missing_idx = set(expected_values.index) - set(actual_values.index)
    if len(missing_idx) == 1:
        raise KeyError(f"Expected {component} #{missing_idx.pop()}, but it is missing {actual_values.index.tolist()}.")
    elif len(missing_idx) > 1:
        raise KeyError(f"Expected {component}s {missing_idx}, but they are missing {actual_values.index.tolist()}.")

    actual_values = actual_values.loc[expected_values.index][mask]
    expected_values = expected_values[mask]

    # Return the result
    return actual_values, expected_values


def extract_extra_info(data: SinglePythonDataset, data_type: str) -> ExtraInfo:
    """
    Reads the dataset and collect all arguments that aren't pgm attributes

    Args:
        data: A dictionary containing all the components
        data_type: The pgm data type; this determines which attributes are considered pgm attributes

    Returns: A dictionary indexed on the object id and containing the extra info for all the objects.
    """
    extra_info: ExtraInfo = {}
    for component, objects in data.items():
        pgm_attr = set(power_grid_meta_data[data_type][component].dtype.names)
        for obj in objects:
            obj_extra_info = {attr: val for attr, val in obj.items() if attr not in pgm_attr}
            if obj_extra_info:
                extra_info[obj["id"]] = obj_extra_info
    return extra_info


def load_json_single_dataset(file_path: Path, data_type: str) -> Tuple[SingleDataset, ExtraInfo]:
    """
    Loads and parses a json file in the most basic way, without using power_grid_model_io functions.

    Args:
        file_path: The json file path
        data_type: The pgm data type; this determines which attributes are considered pgm attributes

    Returns: A native pgm dataset and an extra info lookup table

    """
    try:
        dataset = json_deserialize_from_file(file_path=file_path)
    except PowerGridSerializationError as error:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                dataset = import_json_data(json_file=file_path, data_type=data_type, ignore_extra=True)
        except PowerGridSerializationError:
            pass
        else:
            error = None
            warnings.warn(
                "Provided file path is in a deprecated format. This is a temporary backwards-compatibility measure. "
                "Please upgrade to use_deprecated_format=False or json_serialize_to_file as soon as possible.",
                DeprecationWarning,
            )
        finally:
            if error is not None:
                raise error

    extra_info = extract_extra_info(json.loads(file_path.read_text(encoding="utf-8")), data_type=data_type)
    return dataset, extra_info


def compare_extra_info(actual: ExtraInfo, expected: ExtraInfo, component: str, obj_ids: List[int]):
    # We'll collect all errors, instead of terminating at the first error
    errors = []

    # Check each object in this component
    for obj_id in obj_ids:
        # If there is no extra_info available in the validation data, just skip this object
        if obj_id not in expected:
            continue

        # If the object doesn't exist in the actual data, that's an error
        if obj_id not in actual:
            errors.append(f"Expected {component} #{obj_id}, but it is missing.")
            continue

        # Now for each extra_info in the validation file, check if it matches the actual extra info
        act = actual[obj_id]
        for key, value in expected[obj_id].items():
            # If the extra_info doesn't exist, that's an error
            if key not in act:
                errors.append(f"Expected extra info '{key}' for {component} #{obj_id}, but it is missing.")

            # If the values don't match, that's an error
            elif act[key] != value:
                errors.append(
                    f"Expected extra info '{key}' for {component} #{obj_id} to be {value}, " f"but it is {act[key]}."
                )

    return errors
