# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Dict, List
from unittest.mock import ANY, MagicMock

import numpy as np
import pytest

from power_grid_model_io.converters.base_converter import BaseConverter


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


def test_abstract_methods():
    with pytest.raises(TypeError, match=r"with abstract methods _parse_data, _serialize_data"):
        BaseConverter()


def test_load_input_data__dict(converter: DummyConverter):
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


def test_get_id__private(converter: DummyConverter):
    # Arrange / Act / Assert
    assert converter._get_id(table="node", key={"a": 1, "b": 2}, name=None) == 0
    assert converter._get_id(table="node", key={"a": 1, "b": 3}, name=None) == 1  # change in values
    assert converter._get_id(table="node", key={"a": 1, "c": 2}, name=None) == 2  # change in index
    assert converter._get_id(table="foo", key={"a": 1, "b": 2}, name=None) == 3  # change in table
    assert converter._get_id(table="node", key={"a": 1, "b": 2}, name="bar") == 4  # change in name
    assert converter._get_id(table="node", key={"a": 1, "b": 2}, name=None) == 0  # duplicate name / indices / values


def test_get_id__public(converter: DummyConverter):
    # Arrange
    converter._get_id(table="node", key={"a": 1, "b": 2}, name=None)

    # Act / Assert
    assert converter.get_id(table="node", key={"a": 1, "b": 2}) == 0

    with pytest.raises(KeyError):
        converter.get_id(table="node", key={"a": 1, "b": 3})


def test_lookup_id(converter: DummyConverter):
    # Arrange
    converter._get_id(table="node", key={"a": 1, "b": 2}, name=None)
    converter._get_id(table="node", key={"a": 1, "b": 3}, name=None)  # change in values
    converter._get_id(table="node", key={"a": 1, "c": 2}, name=None)  # change in index
    converter._get_id(table="foo", key={"a": 1, "b": 2}, name=None)  # change in table
    converter._get_id(table="node", key={"a": 1, "b": 2}, name="bar")  # change in name
    converter._get_id(table="node", key={"a": 1, "b": 2}, name=None)  # duplicate name / indices / values

    # Act / Assert
    assert converter.lookup_id(pgm_id=0) == {"table": "node", "key": {"a": 1, "b": 2}}
    assert converter.lookup_id(pgm_id=1) == {"table": "node", "key": {"a": 1, "b": 3}}
    assert converter.lookup_id(pgm_id=2) == {"table": "node", "key": {"a": 1, "c": 2}}
    assert converter.lookup_id(pgm_id=3) == {"table": "foo", "key": {"a": 1, "b": 2}}
    assert converter.lookup_id(pgm_id=4) == {"table": "node", "name": "bar", "key": {"a": 1, "b": 2}}
