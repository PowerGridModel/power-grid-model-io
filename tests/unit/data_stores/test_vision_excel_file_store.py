# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

from power_grid_model_io.data_stores.vision_excel_file_store import VisionExcelFileStore


@patch("power_grid_model_io.data_stores.excel_file_store.pd.read_excel")
@patch("power_grid_model_io.data_stores.excel_file_store.Path.open", mock_open())
def test_header_rows(read_excel_mock: MagicMock):
    # Arrange
    store = VisionExcelFileStore(file_path=Path("dummy.xlsx"))
    read_excel_mock.return_value = {}

    # Act
    store.load()

    # Assert
    read_excel_mock.assert_called_once()
    assert read_excel_mock.call_args_list[0].kwargs["header"] == [0, 1]
