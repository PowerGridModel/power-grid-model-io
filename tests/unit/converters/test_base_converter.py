# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import logging
from typing import Dict, List
from unittest.mock import ANY, MagicMock

import numpy as np
import pytest

from power_grid_model_io.converters.base_converter import BaseConverter


class DummyConverter(BaseConverter[Dict[str, List[Dict[str, int]]]]):
    def __init__(self, source=None, destination=None, log_level=logging.ERROR):
        super().__init__(source, destination, log_level)

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


def test_abstract_methods():
    with pytest.raises(TypeError, match=r"abstract methods .*_parse_data.* .*_serialize_data.*"):
        BaseConverter()


def test_load_input_data(converter: DummyConverter):
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


def test_load_input_data__no_extra_info(converter: DummyConverter):
    # Arrange
    mock_data = MagicMock()

    # Act
    data, extra_info = converter.load_input_data(data=mock_data, make_extra_info=False)

    # Assert
    converter._parse_data.assert_called_once_with(data=mock_data, data_type="input", extra_info=None)  # type: ignore
    assert extra_info == {}


def test_load_input_data__list(converter: DummyConverter):
    # Arrange
    converter._parse_data.return_value = []  # type: ignore

    # Act & Assers
    with pytest.raises(TypeError, match="can not be batch"):
        converter.load_input_data(data={"node": [{"id": 1}, {"id": 2}]})


def test_load_update_data(converter: DummyConverter):
    # Arrange
    converter._parse_data.return_value = {"foo": 1}  # type: ignore

    # Act
    data = converter.load_update_data(data={"node": [{"id": 1}, {"id": 2}]})

    # Assert
    converter._parse_data.assert_called_once_with(  # type: ignore
        data={"node": [{"id": 1}, {"id": 2}]}, data_type="update", extra_info=None
    )
    assert data == {"foo": 1}


def test_load_sym_output_data(converter: DummyConverter):
    # Arrange
    converter._parse_data.return_value = {"foo": 1}  # type: ignore

    # Act
    data = converter.load_sym_output_data(data={"node": [{"id": 1}, {"id": 2}]})

    # Assert
    converter._parse_data.assert_called_once_with(  # type: ignore
        data={"node": [{"id": 1}, {"id": 2}]}, data_type="sym_output", extra_info=None
    )
    assert data == {"foo": 1}


def test_load_asym_output_data(converter: DummyConverter):
    # Arrange
    converter._parse_data.return_value = {"foo": 1}  # type: ignore

    # Act
    data = converter.load_asym_output_data(data={"node": [{"id": 1}, {"id": 2}]})

    # Assert
    converter._parse_data.assert_called_once_with(  # type: ignore
        data={"node": [{"id": 1}, {"id": 2}]}, data_type="asym_output", extra_info=None
    )
    assert data == {"foo": 1}


def test_load_sc_output_data(converter: DummyConverter):
    # Arrange
    converter._parse_data.return_value = {"foo": 1}  # type: ignore

    # Act
    data = converter.load_sc_output_data(data={"node": [{"id": 1}, {"id": 2}]})

    # Assert
    converter._parse_data.assert_called_once_with(  # type: ignore
        data={"node": [{"id": 1}, {"id": 2}]}, data_type="sc_output", extra_info=None
    )
    assert data == {"foo": 1}


def test_convert_data(converter: DummyConverter):
    # Act
    converter.convert(data={"foo": np.array([1])})

    # Assert
    converter._serialize_data.assert_called_once_with(data={"foo": np.array([1])}, extra_info=None)  # type: ignore


def test_save_data(converter: DummyConverter):
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


def test_load_data(converter: DummyConverter):
    # No data supplied
    with pytest.raises(ValueError, match="No data supplied!"):
        converter._load_data(data=None)

    data = converter._load_data(data={"node": [{"id": 1}, {"id": 2}]})
    assert data == data

    source = MagicMock()
    converter_2 = DummyConverter(source=source)
    converter_2._load_data(data=None)
    source.load.assert_called_once()


def test_base_converter_log_level():
    converter = DummyConverter(log_level=logging.DEBUG)
    assert converter.get_log_level() == logging.DEBUG

    converter = DummyConverter()
    assert converter.get_log_level() == logging.ERROR

    converter.set_log_level(logging.DEBUG)
    assert converter.get_log_level() == logging.DEBUG
