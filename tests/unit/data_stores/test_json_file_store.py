# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from unittest.mock import ANY, MagicMock, mock_open, patch

import pytest

from power_grid_model_io.data_stores.json_file_store import JsonFileStore, StructuredData
from power_grid_model_io.utils.json import JsonEncoder


@pytest.fixture()
def single_data() -> StructuredData:
    return {"node": [{"id": 0, "u_rated": 10.5e3}]}


@pytest.fixture()
def batch_data() -> StructuredData:
    return [{"source": [{"id": 1, "p_specified": 1e6}]}, {"source": [{"id": 1, "p_specified": 2e6}]}]


@pytest.fixture(params=["single_data", "batch_data"])
def data(request) -> StructuredData:
    return request.getfixturevalue(request.param)


def test_json_file_store__constructor():
    # Act / Assert
    JsonFileStore(file_path=Path("input_data.json"))  # No exception
    JsonFileStore(file_path=Path("INPUT_DATA.JSON"))  # No exception
    with pytest.raises(ValueError, match=r"should be a \.json file.*csv"):
        JsonFileStore(file_path=Path("input_data.csv"))


@patch("power_grid_model_io.data_stores.json_file_store.Path.open", mock_open())
@patch("power_grid_model_io.data_stores.json_file_store.JsonFileStore._validate")
@patch("power_grid_model_io.data_stores.json_file_store.json.load")
def test_json_file_store__load(mock_json_load: MagicMock, mock_validate: MagicMock, single_data: StructuredData):
    # Arrange
    fs = JsonFileStore(file_path=Path("input_data.json"))
    mock_json_load.return_value = single_data

    # Act
    data = fs.load()

    # Assert
    mock_json_load.assert_called_once()
    mock_validate.assert_called_once_with(data=data)
    assert data == single_data


@patch("power_grid_model_io.data_stores.json_file_store.Path.open", mock_open())
@patch("power_grid_model_io.data_stores.json_file_store.compact_json_dump")
@patch("power_grid_model_io.data_stores.json_file_store.json.dump")
@patch("power_grid_model_io.data_stores.json_file_store.JsonFileStore._validate")
def test_json_file_store__save(
    mock_validate: MagicMock, mock_json_dump: MagicMock, mock_compact_json_dump: MagicMock, single_data: StructuredData
):
    # Arrange
    fs = JsonFileStore(file_path=Path("output_data.json"))

    # Act
    fs.save(data=single_data)

    # Assert
    mock_validate.assert_called_once_with(data=single_data)
    mock_json_dump.assert_not_called()
    mock_compact_json_dump.assert_called_once_with(single_data, ANY, indent=2, max_level=3)


@patch("power_grid_model_io.data_stores.json_file_store.Path.open", mock_open())
@patch("power_grid_model_io.data_stores.json_file_store.compact_json_dump")
@patch("power_grid_model_io.data_stores.json_file_store.json.dump")
@patch("power_grid_model_io.data_stores.json_file_store.JsonFileStore._validate")
def test_json_file_store__save__batch(
    mock_validate: MagicMock, mock_json_dump: MagicMock, mock_compact_json_dump: MagicMock, batch_data: StructuredData
):
    # Arrange
    fs = JsonFileStore(file_path=Path("output_data.json"))

    # Act
    fs.save(data=batch_data)

    # Assert
    mock_validate.assert_called_once_with(data=batch_data)
    mock_json_dump.assert_not_called()
    mock_compact_json_dump.assert_called_once_with(batch_data, ANY, indent=2, max_level=4)


@patch("power_grid_model_io.data_stores.json_file_store.Path.open", mock_open())
@patch("power_grid_model_io.data_stores.json_file_store.compact_json_dump")
@patch("power_grid_model_io.data_stores.json_file_store.json.dump")
@patch("power_grid_model_io.data_stores.json_file_store.JsonFileStore._validate")
def test_json_file_store__save__custom_indent(
    mock_validate: MagicMock, mock_json_dump: MagicMock, mock_compact_json_dump: MagicMock, single_data: StructuredData
):
    # Arrange
    fs = JsonFileStore(file_path=Path("output_data.json"))
    fs.set_indent(4)

    # Act
    fs.save(data=single_data)

    # Assert
    mock_validate.assert_called_once_with(data=single_data)
    mock_json_dump.assert_not_called()
    mock_compact_json_dump.assert_called_once_with(single_data, ANY, indent=4, max_level=3)


@patch("power_grid_model_io.data_stores.json_file_store.Path.open", mock_open())
@patch("power_grid_model_io.data_stores.json_file_store.compact_json_dump")
@patch("power_grid_model_io.data_stores.json_file_store.json.dump")
@patch("power_grid_model_io.data_stores.json_file_store.JsonFileStore._validate")
def test_json_file_store__save__no_indent(
    mock_validate: MagicMock, mock_json_dump: MagicMock, mock_compact_json_dump: MagicMock, single_data: StructuredData
):
    # Arrange
    fs = JsonFileStore(file_path=Path("output_data.json"))
    fs.set_indent(None)

    # Act
    fs.save(data=single_data)

    # Assert
    mock_validate.assert_called_once_with(data=single_data)
    mock_json_dump.assert_called_once_with(single_data, ANY, indent=None, cls=JsonEncoder)
    mock_compact_json_dump.assert_not_called()


@patch("power_grid_model_io.data_stores.json_file_store.Path.open", mock_open())
@patch("power_grid_model_io.data_stores.json_file_store.compact_json_dump")
@patch("power_grid_model_io.data_stores.json_file_store.json.dump")
@patch("power_grid_model_io.data_stores.json_file_store.JsonFileStore._validate")
def test_json_file_store__save__zero_indent(
    mock_validate: MagicMock, mock_json_dump: MagicMock, mock_compact_json_dump: MagicMock, single_data: StructuredData
):
    # Arrange
    fs = JsonFileStore(file_path=Path("output_data.json"))
    fs.set_indent(0)

    # Act
    fs.save(data=single_data)

    # Assert
    mock_validate.assert_called_once_with(data=single_data)
    mock_json_dump.assert_called_once_with(single_data, ANY, indent=0, cls=JsonEncoder)
    mock_compact_json_dump.assert_not_called()


@patch("power_grid_model_io.data_stores.json_file_store.Path.open", mock_open())
@patch("power_grid_model_io.data_stores.json_file_store.compact_json_dump")
@patch("power_grid_model_io.data_stores.json_file_store.json.dump")
@patch("power_grid_model_io.data_stores.json_file_store.JsonFileStore._validate")
def test_json_file_store__save__not_compact(
    mock_validate: MagicMock, mock_json_dump: MagicMock, mock_compact_json_dump: MagicMock, single_data: StructuredData
):
    # Arrange
    fs = JsonFileStore(file_path=Path("output_data.json"))
    fs.set_compact(False)

    # Act
    fs.save(data=single_data)

    # Assert
    mock_validate.assert_called_once_with(data=single_data)
    mock_json_dump.assert_called_once_with(single_data, ANY, indent=2, cls=JsonEncoder)
    mock_compact_json_dump.assert_not_called()


@patch("power_grid_model_io.data_stores.json_file_store.Path.open", mock_open())
@patch("power_grid_model_io.data_stores.json_file_store.compact_json_dump")
@patch("power_grid_model_io.data_stores.json_file_store.json.dump")
@patch("power_grid_model_io.data_stores.json_file_store.JsonFileStore._validate")
def test_json_file_store__save__not_compact_custom_indent(
    mock_validate: MagicMock, mock_json_dump: MagicMock, mock_compact_json_dump: MagicMock, single_data: StructuredData
):
    # Arrange
    fs = JsonFileStore(file_path=Path("output_data.json"))
    fs.set_indent(4)
    fs.set_compact(False)

    # Act
    fs.save(data=single_data)

    # Assert
    mock_validate.assert_called_once_with(data=single_data)
    mock_json_dump.assert_called_once_with(single_data, ANY, indent=4, cls=JsonEncoder)
    mock_compact_json_dump.assert_not_called()


def test_validate(data: StructuredData):
    # Arrange
    fs = JsonFileStore(file_path=Path("dummy.json"))

    # Act / Assert
    fs._validate(data=data)  # no exception


def test_validate__invalid_type():
    # Arrange
    fs = JsonFileStore(file_path=Path("dummy.json"))

    # Act / Assert
    with pytest.raises(TypeError, match=r"Invalid data type.*JsonFileStore.*set"):
        fs._validate(data={"dummy"})  # type: ignore


def test_validate__invalid_list():
    # Arrange
    fs = JsonFileStore(file_path=Path("dummy.json"))

    # Act / Assert
    with pytest.raises(TypeError, match=r"Invalid data type.*JsonFileStore.*List\[int\]"):
        fs._validate(data=[123, 456])  # type: ignore

    with pytest.raises(TypeError, match=r"Invalid data type.*JsonFileStore.*List\[Union\[float, int, str\]\]"):
        fs._validate(data=["foo", 123, 3.1416, "bar"])  # type: ignore
