# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, call, mock_open, patch

import pandas as pd
import pytest

from power_grid_model_io.data_stores.excel_file_store import ExcelFileStore

PandasExcelData = Dict[str, pd.DataFrame]


@pytest.fixture()
def objects_excel() -> PandasExcelData:
    return {
        "Nodes": pd.DataFrame([["A", 111.0], ["B", 222.0]], columns=("NAME", "U_NOM")),
        "Lines": pd.DataFrame([["C", "A", "B", "X"]], columns=("NAME", "NODE0", "NODE1", "TYPE")),
    }


@pytest.fixture()
def specs_excel() -> PandasExcelData:
    return {
        "Colors": pd.DataFrame([["A", "Red"], ["B", "Yellow"]], columns=("NODE", "COLOR")),
        "Lines": pd.DataFrame([["X", 123.0]], columns=("LINE", "R")),
    }


def noop(data: pd.DataFrame, *_args, **_kwargs) -> pd.DataFrame:
    return data


def test_excel_file_store__constructor():
    # Arrange / Act
    fs = ExcelFileStore()  # no exception

    # Assert
    assert fs.files() == {}


def test_excel_file_store__constructor__arg():
    # Arrange / Act
    fs = ExcelFileStore(Path("A.xlsx"))

    # Assert
    assert fs.files() == {"": Path("A.xlsx")}


def test_excel_file_store__constructor__arg_kwargs():
    # Arrange / Act
    fs = ExcelFileStore(Path("A.xlsx"), foo=Path("B.xlsx"), bar=Path("C.xls"))

    # Assert
    assert fs.files() == {"": Path("A.xlsx"), "foo": Path("B.xlsx"), "bar": Path("C.xls")}


def test_excel_file_store__constructor__kwargs():
    # Arrange / Act
    fs = ExcelFileStore(foo=Path("A.xlsx"), bar=Path("B.xls"))

    # Assert
    assert fs.files() == {"foo": Path("A.xlsx"), "bar": Path("B.xls")}


def test_excel_file_store__constructor__too_many_args():
    # Too many (> 1) unnamed arguments
    with pytest.raises(TypeError, match=r"1 to 2.*positional arguments.*3.*given"):
        ExcelFileStore(Path("A.xlsx"), Path("B.xls"))  # type: ignore


def test_excel_file_store__constructor__invalid_main_file():
    with pytest.raises(ValueError, match=r"Excel.*\.docx"):
        ExcelFileStore(Path("A.docx"))


def test_excel_file_store__constructor__invalid_named_file():
    with pytest.raises(ValueError, match=r"Extra.*\.docx"):
        ExcelFileStore(Path("A.xlsx"), extra=Path("B.docx"))


def test_excel_file_store__files__read_only():
    # Arrange
    fs = ExcelFileStore(Path("A.xlsx"), extra=Path("B.xlsx"))
    files = fs.files()

    # Act
    files["extra"] = Path("C.xlsx")

    # Assert
    assert files == {"": Path("A.xlsx"), "extra": Path("C.xlsx")}
    assert fs.files() == {"": Path("A.xlsx"), "extra": Path("B.xlsx")}


@patch("power_grid_model_io.data_stores.excel_file_store.ExcelFileStore._handle_duplicate_columns")
@patch("power_grid_model_io.data_stores.excel_file_store.ExcelFileStore._remove_unnamed_column_placeholders")
@patch("power_grid_model_io.data_stores.excel_file_store.Path.open", mock_open())
@patch("power_grid_model_io.data_stores.excel_file_store.pd.read_excel")
def test_excel_file_store__load(
    mock_re: MagicMock, mock_euci: MagicMock, mock_hdc: MagicMock, objects_excel: PandasExcelData
):
    # Arrange
    fs = ExcelFileStore(file_path=Path("input_data.xlsx"))
    mock_re.return_value = objects_excel
    mock_euci.side_effect = noop
    mock_hdc.side_effect = noop

    # Act
    data = fs.load()

    # Assert
    mock_re.assert_called_once()
    assert mock_euci.call_args_list[0] == call(data=objects_excel["Nodes"], sheet_name="Nodes")
    assert mock_euci.call_args_list[1] == call(data=objects_excel["Lines"], sheet_name="Lines")
    assert mock_hdc.call_args_list[0] == call(data=objects_excel["Nodes"], sheet_name="Nodes")
    assert mock_hdc.call_args_list[1] == call(data=objects_excel["Lines"], sheet_name="Lines")
    pd.testing.assert_frame_equal(data["Nodes"], objects_excel["Nodes"])
    pd.testing.assert_frame_equal(data["Lines"], objects_excel["Lines"])


@patch("power_grid_model_io.data_stores.excel_file_store.ExcelFileStore._handle_duplicate_columns")
@patch("power_grid_model_io.data_stores.excel_file_store.ExcelFileStore._remove_unnamed_column_placeholders")
@patch("power_grid_model_io.data_stores.excel_file_store.Path.open", mock_open())
@patch("power_grid_model_io.data_stores.excel_file_store.pd.read_excel")
def test_excel_file_store__load__extra(
    mock_re: MagicMock,
    mock_euci: MagicMock,
    mock_hdc: MagicMock,
    objects_excel: PandasExcelData,
    specs_excel: PandasExcelData,
):
    # Arrange
    fs = ExcelFileStore(Path("input_data.xlsx"), foo=Path("foo_types.xlsx"))
    mock_re.side_effect = (objects_excel, specs_excel)
    mock_euci.side_effect = noop
    mock_hdc.side_effect = noop

    # Act
    data = fs.load()

    # Assert
    assert mock_re.call_count == 2
    assert mock_euci.call_args_list[0] == call(data=objects_excel["Nodes"], sheet_name="Nodes")
    assert mock_euci.call_args_list[1] == call(data=objects_excel["Lines"], sheet_name="Lines")
    assert mock_euci.call_args_list[2] == call(data=specs_excel["Colors"], sheet_name="Colors")
    assert mock_euci.call_args_list[3] == call(data=specs_excel["Lines"], sheet_name="Lines")
    assert mock_hdc.call_args_list[0] == call(data=objects_excel["Nodes"], sheet_name="Nodes")
    assert mock_hdc.call_args_list[1] == call(data=objects_excel["Lines"], sheet_name="Lines")
    assert mock_hdc.call_args_list[2] == call(data=specs_excel["Colors"], sheet_name="Colors")
    assert mock_hdc.call_args_list[3] == call(data=specs_excel["Lines"], sheet_name="Lines")
    pd.testing.assert_frame_equal(data["Nodes"], objects_excel["Nodes"])
    pd.testing.assert_frame_equal(data["Lines"], objects_excel["Lines"])
    pd.testing.assert_frame_equal(data["foo.Colors"], specs_excel["Colors"])
    pd.testing.assert_frame_equal(data["foo.Lines"], specs_excel["Lines"])


@patch("power_grid_model_io.data_stores.excel_file_store.ExcelFileStore._handle_duplicate_columns")
@patch("power_grid_model_io.data_stores.excel_file_store.ExcelFileStore._remove_unnamed_column_placeholders")
@patch("power_grid_model_io.data_stores.excel_file_store.Path.open", mock_open())
@patch("power_grid_model_io.data_stores.excel_file_store.pd.read_excel")
def test_excel_file_store__load__extra__duplicate_sheet_name(
    mock_re: MagicMock, mock_euci: MagicMock, mock_hdc: MagicMock
):
    # Arrange
    foo_data = {"bar.Nodes": pd.DataFrame()}
    bar_data = {"Nodes": pd.DataFrame()}
    fs = ExcelFileStore(Path("foo.xlsx"), bar=Path("bar.xlsx"))
    mock_re.side_effect = (foo_data, bar_data)
    mock_euci.side_effect = noop
    mock_hdc.side_effect = noop

    # Act / Assert
    with pytest.raises(ValueError, match=r"Duplicate sheet name.+bar\.Nodes"):
        fs.load()
