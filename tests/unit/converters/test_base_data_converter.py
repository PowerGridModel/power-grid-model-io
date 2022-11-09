# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Dict, List
from unittest.mock import ANY, MagicMock

import numpy as np
import pytest

from power_grid_model_io.converters.base_converter import BaseConverter
from power_grid_model_io.data_stores.excel_file_store import ExcelFileStore


class DummyConverter(BaseConverter[Dict[str, List[Dict[str, int]]]]):
    def _parse_data(self, data, data_type, extra_info=None):
        # No need to implement _parse_data() for testing purposes
        pass

    def _serialize_data(self, data, extra_info=None):
        # No need to implement _serialize_data() for testing purposes
        pass


@pytest.fixture
def converter():
    converter = DummyConverter()
    converter._parse_data = MagicMock()
    converter._serialize_data = MagicMock()
    converter._serialize_data.return_value = {"node": [{"id": 1}, {"id": 2}]}
    return converter


def test_base_converter__abstract_methods():
    with pytest.raises(TypeError, match=r"with abstract methods _parse_data, _serialize_data"):
        BaseConverter()


def test_converter__load_input_data__dict(converter: DummyConverter):
    # Arrange
    def add_extra_info(data, data_type, extra_info):
        extra_info[1] = "Foo"
        return {"foo": 1}

    converter._parse_data.side_effect = add_extra_info  # type: ignore

    # Act
    data, extra_info = converter.load_input_data(data={"node": [{"id": 1}, {"id": 2}]})

    # Assert
    converter._parse_data.assert_called_once_with(  # type: ignore
        data={"node": [{"id": 1}, {"id": 2}]}, data_type="input", extra_info=ANY
    )
    assert data == {"foo": 1}
    assert extra_info == {1: "Foo"}


def test_converter__load_input_data__list(converter: DummyConverter):
    # Arrange
    converter._parse_data.return_value = []  # type: ignore

    # Act & Assers
    with pytest.raises(TypeError, match="can not be batch"):
        converter.load_input_data(data={"node": [{"id": 1}, {"id": 2}]})


def test_converter__load_update_data(converter: DummyConverter):
    # Arrange
    converter._parse_data.return_value = {"foo": 1}  # type: ignore

    # Act
    data = converter.load_update_data(data={"node": [{"id": 1}, {"id": 2}]})

    # Assert
    converter._parse_data.assert_called_once_with(  # type: ignore
        data={"node": [{"id": 1}, {"id": 2}]}, data_type="update", extra_info=None
    )
    assert data == {"foo": 1}


def test_converter__load_sym_output_data(converter: DummyConverter):
    # Arrange
    converter._parse_data.return_value = {"foo": 1}  # type: ignore

    # Act
    data = converter.load_sym_output_data(data={"node": [{"id": 1}, {"id": 2}]})

    # Assert
    converter._parse_data.assert_called_once_with(  # type: ignore
        data={"node": [{"id": 1}, {"id": 2}]}, data_type="sym_output", extra_info=None
    )
    assert data == {"foo": 1}


def test_converter__load_asym_output_data(converter: DummyConverter):
    # Arrange
    converter._parse_data.return_value = {"foo": 1}  # type: ignore

    # Act
    data = converter.load_asym_output_data(data={"node": [{"id": 1}, {"id": 2}]})

    # Assert
    converter._parse_data.assert_called_once_with(  # type: ignore
        data={"node": [{"id": 1}, {"id": 2}]}, data_type="asym_output", extra_info=None
    )
    assert data == {"foo": 1}


def test_converter__convert_data(converter: DummyConverter):
    # Act
    converter.convert(data={"foo": np.array([1])})

    # Assert
    converter._serialize_data.assert_called_once_with(data={"foo": np.array([1])}, extra_info=None)  # type: ignore


def test_converter__save_data(converter: DummyConverter):
    # No destination supplied
    with pytest.raises(ValueError, match="No destination supplied!"):
        converter.save(data={"foo": np.array([1])})

    # Destination supplied as argument
    destination = MagicMock()
    converter.save(data={"foo": np.array([1])}, destination=destination)
    destination.save.assert_called_once_with(data={"node": [{"id": 1}, {"id": 2}]})

    # Destination supplied at instantiation
    destination2 = MagicMock()
    converter_2 = DummyConverter(destination=destination2)
    converter_2.save(data={"foo": np.array([1])})
    destination2.save.assert_called_once()


def test_converter__load_data(converter: DummyConverter):
    # No data supplied
    with pytest.raises(ValueError, match="No data supplied!"):
        converter._load_data(data=None)

    data = converter._load_data(data={"node": [{"id": 1}, {"id": 2}]})
    assert data == data

    source = MagicMock()
    converter_2 = DummyConverter(source=source)
    converter_2._load_data(data=None)
    source.load.assert_called_once()
