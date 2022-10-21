# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import Dict

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


# TODO Write similar unittests as for json data
# @patch("power_grid_model_io.data_stores.json_file_store.Path.open", mock_open())
# @patch("power_grid_model_io.data_stores.json_file_store.JsonFileStore._validate")
# @patch("power_grid_model_io.data_stores.json_file_store.pd.read_excel")
# def test_excel_file_store__load(mock_read_excel: MagicMock, mock_validate: MagicMock, objects_excel: PandasExcelData):
#         # Arrange
#         fs = ExcelFileStore(file_path=Path("input_data.json"))
#         mock_json_load.return_value = single_data
#
#         # Act
#         data = fs.load()

# Assert
# mock_json_load.assert_called_once()
# mock_validate.assert_called_once_with(data=data)
# assert data == single_data
