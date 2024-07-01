# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import json
import logging
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytest
from power_grid_model.data_types import SingleDataset

from power_grid_model_io.converters import VisionExcelConverter
from power_grid_model_io.converters.vision_excel_converter import CONFIG_PATH
from power_grid_model_io.data_stores.base_data_store import DICT_KEY_NUMBER, LANGUAGE_EN, VISION_EXCEL_LAN_DICT
from power_grid_model_io.data_types import ExtraInfo
from power_grid_model_io.utils.json import JsonEncoder
from power_grid_model_io.utils.uuid_excel_cvtr import convert_guid_vision_excel

from ..utils import compare_extra_info, component_attributes, component_objects, load_json_single_dataset, select_values

DATA_PATH = Path(__file__).parents[2] / "data" / "vision"
SOURCE_FILE = DATA_PATH / "vision_{language:s}.xlsx"
SOURCE_FILE_97 = DATA_PATH / "vision_{language:s}_9_7.xlsx"
VALIDATION_FILE = DATA_PATH / "pgm_input_data_{language:s}.json"
LANGUAGES = ["en", "nl"]
LANGUAGES_97 = ["en"]
VALIDATION_EN = Path(str(VALIDATION_FILE).format(language="en"))
CUSTOM_MAPPING_FILE = DATA_PATH / "vision_9_5_{language:s}.yaml"
VISION_97_MAPPING_FILE = CONFIG_PATH / "vision_en_9_7.yaml"
terms_changed = {"Grounding1": "N1", "Grounding2": "N2", "Grounding3": "N3", "Load.Behaviour": "Behaviour"}


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


def test_input_data_custom_yaml():
    """
    Unit test to preload the expected and actual data, using a different mapping file other than the one in the default location
    """
    for language in LANGUAGES:
        # Arrange
        actual, _ = VisionExcelConverter(
            Path(str(SOURCE_FILE).format(language=language)),
            language=language,
            mapping_file=Path(str(CUSTOM_MAPPING_FILE).format(language=language)),
        ).load_input_data()
        expected, _ = load_validation_data(language=language)
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
    ("language", "table", "name", "columns", "filtering_mask"),
    [
        ("en", "Transformer loads", "transformer", ["Node.Number", "Subnumber"], None),
        ("en", "Transformer loads", "internal_node", ["Node.Number", "Subnumber"], None),
        ("en", "Transformer loads", "load", ["Node.Number", "Subnumber"], None),
        ("en", "Transformer loads", "generation", ["Node.Number", "Subnumber"], np.array([False, True])),
        ("en", "Transformer loads", "pv_generation", ["Node.Number", "Subnumber"], np.array([False, True])),
        ("nl", "Transformatorbelastingen", "transformer", ["Knooppunt.Nummer", "Subnummer"], None),
        ("nl", "Transformatorbelastingen", "internal_node", ["Knooppunt.Nummer", "Subnummer"], None),
        ("nl", "Transformatorbelastingen", "load", ["Knooppunt.Nummer", "Subnummer"], None),
        ("nl", "Transformatorbelastingen", "generation", ["Knooppunt.Nummer", "Subnummer"], np.array([False, True])),
        ("nl", "Transformatorbelastingen", "pv_generation", ["Knooppunt.Nummer", "Subnummer"], np.array([False, True])),
    ],
)
def test_get_get_virtual_id(
    language: str, table: str, name: str, columns: List[str], filtering_mask: Optional[np.ndarray]
):
    # Arrange
    converter = vision_excel_converter(language=language)
    _, extra_info = load_and_convert_excel_file(language=language)

    assert converter._source is not None
    source_data = converter._source.load()[table][columns]
    if filtering_mask is not None:
        source_data = source_data[filtering_mask]

    # Act/Assert
    assert isinstance(source_data, pd.DataFrame)
    for _, (node_number, sub_number) in source_data.iterrows():
        pgm_id = converter.get_virtual_id(table=table, obj_name=name, node_number=node_number, sub_number=sub_number)
        assert extra_info[pgm_id]["id_reference"] == {
            "table": table,
            "name": name,
            "key": {columns[0]: node_number, columns[1]: sub_number},
        }


def test_log_levels(capsys):
    """Test the log levels of the VisionExcelConverter class.
    The VisionExcelConverter made heavy use of tabular converter, which contains several debug log in their code
    This test is to ensure that the log level of VisionExcelConverter is passed to TabularConverter so that we could
    expect a more uniform performance of the log level across the converters.
    """
    cvtr1 = VisionExcelConverter()
    cvtr1.set_log_level(logging.WARNING)
    assert cvtr1.get_log_level() == logging.WARNING
    cvtr2 = VisionExcelConverter()
    cvtr2.set_log_level(logging.ERROR)
    assert cvtr1.get_log_level() == logging.WARNING
    assert cvtr2.get_log_level() == logging.ERROR
    cvtr3 = VisionExcelConverter()
    cvtr3.set_log_level(logging.DEBUG)
    assert cvtr1.get_log_level() == logging.WARNING
    assert cvtr2.get_log_level() == logging.ERROR
    assert cvtr3.get_log_level() == logging.DEBUG
    cvtr4 = VisionExcelConverter()
    cvtr4.set_log_level(logging.CRITICAL)
    assert cvtr1.get_log_level() == logging.WARNING
    assert cvtr2.get_log_level() == logging.ERROR
    assert cvtr3.get_log_level() == logging.DEBUG
    assert cvtr4.get_log_level() == logging.CRITICAL

    source_file = Path(str(SOURCE_FILE).format(language="en"))
    cvtr5 = VisionExcelConverter(source_file, language="en")
    cvtr5.set_log_level(logging.CRITICAL)
    outerr = capsys.readouterr()
    assert "debug" not in outerr.out


def prep_vision_97(language: str) -> VisionExcelConverter:
    source_file = Path(str(SOURCE_FILE_97).format(language=LANGUAGE_EN))
    return VisionExcelConverter(
        source_file, language="en", mapping_file=VISION_97_MAPPING_FILE, terms_changed=terms_changed
    )


def test_uuid_excel_input():
    ref_file_97 = convert_guid_vision_excel(
        excel_file=Path(str(SOURCE_FILE_97).format(language=LANGUAGE_EN)),
        number=VISION_EXCEL_LAN_DICT[LANGUAGE_EN][DICT_KEY_NUMBER],
        terms_changed=terms_changed,
    )
    data_convtd, _ = VisionExcelConverter(
        source_file=ref_file_97, mapping_file=VISION_97_MAPPING_FILE
    ).load_input_data()
    vision_cvtr = prep_vision_97(language=LANGUAGE_EN)
    data_native, _ = vision_cvtr.load_input_data()

    assert len(data_native) == len(data_convtd)


def test_guid_extra_info():
    print("extra_info")
    vision_cvtr = prep_vision_97(language=LANGUAGE_EN)
    _, extra_info = vision_cvtr.load_input_data()

    assert extra_info[0]["GUID"] == "{7FF722ED-33B3-4761-84AC-A164310D3C86}"
    assert extra_info[1]["GUID"] == "{1ED177A7-1F5D-4D81-8DE7-AB3E58512E0B}"
    assert extra_info[2]["GUID"] == "{DDE3457B-DB9A-4DA9-9564-6F49E0F296BD}"
    assert extra_info[3]["GUID"] == "{A79AFDE9-4096-4BEB-AB63-2B851D7FC6D1}"
    assert extra_info[4]["GUID"] == "{7848DBC8-9685-452C-89AF-9AB308224689}"

    for i in range(5, len(extra_info)):
        assert "GUID" not in extra_info[i]
