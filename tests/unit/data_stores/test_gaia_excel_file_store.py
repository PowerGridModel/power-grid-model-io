# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from unittest.mock import MagicMock, Mock, mock_open, patch

from power_grid_model_io.data_stores.gaia_excel_file_store import GaiaExcelFileStore


@patch("power_grid_model_io.data_stores.excel_file_store.pd.read_excel")
@patch("power_grid_model_io.data_stores.excel_file_store.Path.open", mock_open())
def test_header_rows(read_excel_mock: MagicMock):
    # Arrange
    store = GaiaExcelFileStore(file_path=Path("dummy.xlsx"))
    read_excel_mock.return_value = {}

    # Act
    store.load()

    # Assert
    read_excel_mock.assert_called_once()
    assert read_excel_mock.call_args_list[0].kwargs["header"] == [0, 1]


@patch("power_grid_model_io.data_stores.excel_file_store.pd.read_excel")
def test_types_file(read_excel_mock: MagicMock):
    # Arrange
    main_file = MagicMock(suffix=".xlsx")
    types_file = MagicMock(suffix=".xlsx")
    store = GaiaExcelFileStore(file_path=main_file, types_file=types_file)
    read_excel_mock.return_value = {}

    # Act
    store.load()

    # Assert
    main_file.open.assert_called_once()
    types_file.open.assert_called_once()
    assert read_excel_mock.call_count == 2
