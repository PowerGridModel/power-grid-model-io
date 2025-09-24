# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import pandas as pd

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
    assert mock_excel_file.return_value.parse.call_count == 2


@patch("power_grid_model_io.data_stores.excel_file_store.pd.ExcelFile")
@patch("power_grid_model_io.data_stores.excel_file_store.Path.open", mock_open())
def test_name_column_dtype_conversion(mock_excel_file: MagicMock):
    store = VisionExcelFileStore(file_path=Path("dummy.xlsx"))
    mock_excel_file.return_value.sheet_names = ["test_sheet"]

    preview_df = pd.DataFrame(columns=["Mock.Name", "Other.Column", "ID"])

    def mock_parse(*args, **kwargs):
        if kwargs.get("nrows") == 0:
            return preview_df
        else:
            actual_data = {
                "Mock.Name": [12345678900000000000, 987.654],
                "Other.Column": ["value1", "value2"],
                "ID": [1, 2],
                "ratio": [0.1, 0.2],
            }
            df = pd.DataFrame(actual_data)

            if "dtype" in kwargs:
                for col, dtype_val in kwargs["dtype"].items():
                    if col in df.columns and dtype_val is str:
                        df[col] = df[col].apply(lambda x: str(int(x)) if float(x).is_integer() else str(x))

            return df

    mock_excel_file.return_value.parse.side_effect = mock_parse

    data = store.load()
    result_df = data["test_sheet"]

    assert mock_excel_file.return_value.parse.call_count == 2

    first_call = mock_excel_file.return_value.parse.call_args_list[0]
    assert first_call[1]["nrows"] == 0

    second_call = mock_excel_file.return_value.parse.call_args_list[1]
    assert "dtype" in second_call[1]
    assert "Mock.Name" in second_call[1]["dtype"]
    assert second_call[1]["dtype"]["Mock.Name"] is str

    assert result_df["Mock.Name"][0] == "12345678900000000000"  # Long int as string
    assert result_df["Mock.Name"][1] == "987.654"  # Float as string
    assert result_df["Other.Column"][0] == "value1"
    assert result_df["Other.Column"][1] == "value2"
    assert result_df["ID"][0] == 1
    assert result_df["ID"][1] == 2
    assert result_df["ratio"][0] == 0.1
    assert result_df["ratio"][1] == 0.2
