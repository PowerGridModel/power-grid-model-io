# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import json
from functools import lru_cache
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import pytest
from power_grid_model.data_types import SingleDataset

from power_grid_model_io.converters import VisionExcelConverter
from power_grid_model_io.data_types import ExtraInfo
from power_grid_model_io.utils.json import JsonEncoder

from ..utils import compare_extra_info, component_attributes, component_objects, load_json_single_dataset, select_values

DATA_PATH = Path(__file__).parents[2] / "data" / "vision"
SOURCE_FILE = DATA_PATH / "vision_{language:s}.xlsx"
VALIDATION_FILE = DATA_PATH / "pgm_input_data_{language:s}.json"
LANGUAGES = ["en", "nl"]
VALIDATION_EN = Path(str(VALIDATION_FILE).format(language="en"))


@lru_cache
def vision_excel_converter(language: str) -> VisionExcelConverter:
    """
    Read the excel file
    """
    source_file = Path(str(SOURCE_FILE).format(language=language))
    return VisionExcelConverter(source_file, language=language)


@lru_cache
def load_and_convert_excel_file(language: str) -> Tuple[SingleDataset, ExtraInfo]:
    """
    Convert the excel file
    """
    data, extra_info = vision_excel_converter(language=language).load_input_data()
    return data, extra_info


@lru_cache
def load_validation_data(language: str) -> Tuple[SingleDataset, ExtraInfo]:
    """
    Load the validation data from the json file
    """
    validation_file = Path(str(VALIDATION_FILE).format(language=language))
    data, extra_info = load_json_single_dataset(validation_file, data_type="input")
    return data, extra_info


@pytest.fixture
def input_data(request) -> Tuple[SingleDataset, SingleDataset]:
    """
    Read the excel file and the json file, and return the input_data
    """
    actual, _ = load_and_convert_excel_file(language=request.param)
    expected, _ = load_validation_data(language=request.param)
    return actual, expected


@pytest.fixture
def extra_info(request) -> Tuple[ExtraInfo, ExtraInfo]:
    """
    Read the excel file and the json file, and return the extra_info
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


@pytest.mark.parametrize(("component", "attribute"), component_attributes(VALIDATION_EN, data_type="input"))
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
def test_extra_info(extra_info: Tuple[ExtraInfo, ExtraInfo], component: str, obj_ids: List[int]):
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


@pytest.mark.parametrize("extra_info", LANGUAGES, indirect=True)
def test_extra_info__serializable(extra_info):
    # Arrange
    actual, _ = extra_info

    # Assert
    json.dumps(actual, cls=JsonEncoder)  # expect no exception


@pytest.mark.parametrize(("language", "table", "column"), [("en", "Nodes", "Number"), ("nl", "Knooppunten", "Nummer")])
def test_get_node_id(language: str, table: str, column: str):
    # Arrange
    converter = vision_excel_converter(language=language)
    _, extra_info = load_and_convert_excel_file(language=language)

    assert converter._source is not None
    source_data = converter._source.load()[table][column]

    # Act/Assert
    for number in source_data:
        pgm_id = converter.get_node_id(number=number)
        assert extra_info[pgm_id]["id_reference"] == {"table": table, "key": {column: number}}


@pytest.mark.parametrize(
    ("language", "table", "column"),
    [
        ("en", "Cables", "Number"),
        ("en", "Lines", "Number"),
        ("en", "Reactance coils", "Number"),
        ("en", "Links", "Number"),
        ("en", "Transformers", "Number"),
        ("en", "Special transformers", "Number"),
        ("en", "Three winding transformers", "Number"),
        ("nl", "Kabels", "Nummer"),
        ("nl", "Verbindingen", "Nummer"),
        ("nl", "Smoorspoelen", "Nummer"),
        ("nl", "Links", "Nummer"),
        ("nl", "Transformatoren", "Nummer"),
        ("nl", "Speciale transformatoren", "Nummer"),
        ("nl", "Driewikkelingstransformatoren", "Nummer"),
    ],
)
def test_get_branch_id(language: str, table: str, column: str):
    # Arrange
    converter = vision_excel_converter(language=language)
    _, extra_info = load_and_convert_excel_file(language=language)

    assert converter._source is not None
    source_data = converter._source.load()[table][column]

    # Act/Assert
    for number in source_data:
        pgm_id = converter.get_branch_id(table=table, number=number)
        assert extra_info[pgm_id]["id_reference"] == {"table": table, "key": {column: number}}


@pytest.mark.parametrize(
    ("language", "table", "columns"),
    [
        ("en", "Loads", ["Node.Number", "Subnumber"]),
        ("en", "Synchronous generators", ["Node.Number", "Subnumber"]),
        ("en", "Wind turbines", ["Node.Number", "Subnumber"]),
        ("en", "Pvs", ["Node.Number", "Subnumber"]),
        ("en", "Sources", ["Node.Number", "Subnumber"]),
        ("en", "Zigzag transformers", ["Node.Number", "Subnumber"]),
        ("en", "Capacitors", ["Node.Number", "Subnumber"]),
        ("en", "Reactors", ["Node.Number", "Subnumber"]),
        ("nl", "Belastingen", ["Knooppunt.Nummer", "Subnummer"]),
        ("nl", "Synchrone generatoren", ["Knooppunt.Nummer", "Subnummer"]),
        ("nl", "Windturbines", ["Knooppunt.Nummer", "Subnummer"]),
        ("nl", "Pv's", ["Knooppunt.Nummer", "Subnummer"]),
        ("nl", "Netvoedingen", ["Knooppunt.Nummer", "Subnummer"]),
        ("nl", "Nulpuntstransformatoren", ["Knooppunt.Nummer", "Subnummer"]),
        ("nl", "Condensatoren", ["Knooppunt.Nummer", "Subnummer"]),
        ("nl", "Spoelen", ["Knooppunt.Nummer", "Subnummer"]),
    ],
)
def test_get_get_appliance_id(language: str, table: str, columns: List[str]):
    # Arrange
    converter = vision_excel_converter(language=language)
    _, extra_info = load_and_convert_excel_file(language=language)

    assert converter._source is not None
    source_data = converter._source.load()[table][columns]

    # Act/Assert
    assert isinstance(source_data, pd.DataFrame)
    for _, (node_number, sub_number) in source_data.iterrows():
        pgm_id = converter.get_appliance_id(table=table, node_number=node_number, sub_number=sub_number)
        assert extra_info[pgm_id]["id_reference"] == {
            "table": table,
            "key": {columns[0]: node_number, columns[1]: sub_number},
        }


@pytest.mark.parametrize(
    ("language", "table", "name", "columns"),
    [
        ("en", "Transformer loads", "transformer", ["Node.Number", "Subnumber"]),
        ("en", "Transformer loads", "internal_node", ["Node.Number", "Subnumber"]),
        ("en", "Transformer loads", "load", ["Node.Number", "Subnumber"]),
        ("en", "Transformer loads", "generation", ["Node.Number", "Subnumber"]),
        ("en", "Transformer loads", "pv_generation", ["Node.Number", "Subnumber"]),
        ("nl", "Transformatorbelastingen", "transformer", ["Knooppunt.Nummer", "Subnummer"]),
        ("nl", "Transformatorbelastingen", "internal_node", ["Knooppunt.Nummer", "Subnummer"]),
        ("nl", "Transformatorbelastingen", "load", ["Knooppunt.Nummer", "Subnummer"]),
        ("nl", "Transformatorbelastingen", "generation", ["Knooppunt.Nummer", "Subnummer"]),
        ("nl", "Transformatorbelastingen", "pv_generation", ["Knooppunt.Nummer", "Subnummer"]),
    ],
)
def test_get_get_virtual_id(language: str, table: str, name: str, columns: List[str]):
    # Arrange
    converter = vision_excel_converter(language=language)
    _, extra_info = load_and_convert_excel_file(language=language)

    assert converter._source is not None
    source_data = converter._source.load()[table][columns]

    # Act/Assert
    assert isinstance(source_data, pd.DataFrame)
    for _, (node_number, sub_number) in source_data.iterrows():
        pgm_id = converter.get_virtual_id(table=table, obj_name=name, node_number=node_number, sub_number=sub_number)
        assert extra_info[pgm_id]["id_reference"] == {
            "table": table,
            "name": name,
            "key": {columns[0]: node_number, columns[1]: sub_number},
        }
