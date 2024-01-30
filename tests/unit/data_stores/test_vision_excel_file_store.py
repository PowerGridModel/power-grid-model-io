# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from power_grid_model_io.data_stores.vision_excel_file_store import VisionExcelFileStore


@patch("power_grid_model_io.data_stores.excel_file_store.pd.ExcelFile")
@patch("power_grid_model_io.data_stores.excel_file_store.Path.open", mock_open())
def test_header_rows(mock_excel_file: MagicMock):
    # Arrange
    store = VisionExcelFileStore(file_path=Path("dummy.xlsx"))
    mock_excel_file.return_value.sheet_names = ["foo"]

    # Act
    data = store.load()
    data["foo"]

    # Assert
    mock_excel_file.return_value.parse.assert_called_once_with("foo", header=[0, 1])
