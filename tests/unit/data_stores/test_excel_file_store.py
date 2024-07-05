# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import Dict
from unittest.mock import MagicMock, call, mock_open, patch

import numpy as np
import pandas as pd
import pytest
from pytest import param
from structlog.testing import capture_logs

from power_grid_model_io.data_stores.excel_file_store import ExcelFileStore
from power_grid_model_io.data_types.tabular_data import TabularData

from ...utils import MockExcelFile, assert_log_exists

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


def test_constructor():
    # Arrange / Act
    fs = ExcelFileStore()  # no exception

    # Assert
    assert fs.files() == {}


def test_constructor__arg():
    # Arrange / Act
    fs = ExcelFileStore(Path("A.xlsx"))

    # Assert
    assert fs.files() == {"": Path("A.xlsx")}


def test_constructor__arg_kwargs():
    # Arrange / Act
    fs = ExcelFileStore(Path("A.xlsx"), foo=Path("B.xlsx"), bar=Path("C.xls"))

    # Assert
    assert fs.files() == {"": Path("A.xlsx"), "foo": Path("B.xlsx"), "bar": Path("C.xls")}


def test_constructor__kwargs():
    # Arrange / Act
    fs = ExcelFileStore(foo=Path("A.xlsx"), bar=Path("B.xls"))

    # Assert
    assert fs.files() == {"foo": Path("A.xlsx"), "bar": Path("B.xls")}


def test_constructor__too_many_args():
    # Too many (> 1) unnamed arguments
    with pytest.raises(TypeError, match=r"1 to 2.*positional arguments.*3.*given"):
        ExcelFileStore(Path("A.xlsx"), Path("B.xls"))  # type: ignore


def test_constructor__invalid_main_file():
    with pytest.raises(ValueError, match=r"Excel.*\.docx"):
        ExcelFileStore(Path("A.docx"))


def test_constructor__invalid_named_file():
    with pytest.raises(ValueError, match=r"Extra.*\.docx"):
        ExcelFileStore(Path("A.xlsx"), extra=Path("B.docx"))


def test_files__read_only():
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
@patch("power_grid_model_io.data_stores.excel_file_store.pd.ExcelFile")
def test_load(
    mock_excel_file: MagicMock,
    mock_remove_unnamed_column_placeholders: MagicMock,
    mock_handle_duplicate_columns: MagicMock,
    objects_excel: PandasExcelData,
):
    fs = ExcelFileStore(file_path=Path("input_data.xlsx"))
    mock_excel_file.return_value = MockExcelFile(objects_excel)
    mock_remove_unnamed_column_placeholders.side_effect = noop
    mock_handle_duplicate_columns.side_effect = noop

    # Act
    data = fs.load()

    # Assert
    mock_excel_file.assert_called_once()
    pd.testing.assert_frame_equal(data["Nodes"], objects_excel["Nodes"])
    pd.testing.assert_frame_equal(data["Lines"], objects_excel["Lines"])
    assert mock_remove_unnamed_column_placeholders.call_args_list[0] == call(data=objects_excel["Nodes"])
    assert mock_remove_unnamed_column_placeholders.call_args_list[1] == call(data=objects_excel["Lines"])
    assert mock_handle_duplicate_columns.call_args_list[0] == call(data=objects_excel["Nodes"], sheet_name="Nodes")
    assert mock_handle_duplicate_columns.call_args_list[1] == call(data=objects_excel["Lines"], sheet_name="Lines")


@patch("power_grid_model_io.data_stores.excel_file_store.ExcelFileStore._handle_duplicate_columns")
@patch("power_grid_model_io.data_stores.excel_file_store.ExcelFileStore._remove_unnamed_column_placeholders")
@patch("power_grid_model_io.data_stores.excel_file_store.pd.ExcelFile")
def test_load__extra(
    mock_excel_file: MagicMock,
    mock_remove_unnamed_column_placeholders: MagicMock,
    mock_handle_duplicate_columns: MagicMock,
    objects_excel: PandasExcelData,
    specs_excel: PandasExcelData,
):
    # Arrange
    fs = ExcelFileStore(Path("input_data.xlsx"), foo=Path("foo_types.xlsx"))
    mock_excel_file.side_effect = (MockExcelFile(objects_excel), MockExcelFile(specs_excel))
    mock_remove_unnamed_column_placeholders.side_effect = noop
    mock_handle_duplicate_columns.side_effect = noop

    # Act
    data = fs.load()

    # Assert
    assert mock_excel_file.call_count == 2
    pd.testing.assert_frame_equal(data["Nodes"], objects_excel["Nodes"])
    pd.testing.assert_frame_equal(data["Lines"], objects_excel["Lines"])
    pd.testing.assert_frame_equal(data["foo.Colors"], specs_excel["Colors"])
    pd.testing.assert_frame_equal(data["foo.Lines"], specs_excel["Lines"])
    assert mock_remove_unnamed_column_placeholders.call_args_list[0] == call(data=objects_excel["Nodes"])
    assert mock_remove_unnamed_column_placeholders.call_args_list[1] == call(data=objects_excel["Lines"])
    assert mock_remove_unnamed_column_placeholders.call_args_list[2] == call(data=specs_excel["Colors"])
    assert mock_remove_unnamed_column_placeholders.call_args_list[3] == call(data=specs_excel["Lines"])
    assert mock_handle_duplicate_columns.call_args_list[0] == call(data=objects_excel["Nodes"], sheet_name="Nodes")
    assert mock_handle_duplicate_columns.call_args_list[1] == call(data=objects_excel["Lines"], sheet_name="Lines")
    assert mock_handle_duplicate_columns.call_args_list[2] == call(data=specs_excel["Colors"], sheet_name="Colors")
    assert mock_handle_duplicate_columns.call_args_list[3] == call(data=specs_excel["Lines"], sheet_name="Lines")


@patch("power_grid_model_io.data_stores.excel_file_store.ExcelFileStore._handle_duplicate_columns")
@patch("power_grid_model_io.data_stores.excel_file_store.ExcelFileStore._remove_unnamed_column_placeholders")
@patch("power_grid_model_io.data_stores.excel_file_store.pd.ExcelFile")
def test_load__extra__duplicate_sheet_name(
    mock_excel_file: MagicMock,
    mock_remove_unnamed_column_placeholders: MagicMock,
    mock_handle_duplicate_columns: MagicMock,
):
    # Arrange
    foo_data = {"bar.Nodes": pd.DataFrame()}
    bar_data = {"Nodes": pd.DataFrame()}
    fs = ExcelFileStore(Path("foo.xlsx"), bar=Path("bar.xlsx"))
    mock_excel_file.side_effect = (MockExcelFile(foo_data), MockExcelFile(bar_data))
    mock_remove_unnamed_column_placeholders.side_effect = noop
    mock_handle_duplicate_columns.side_effect = noop

    # Act / Assert
    with pytest.raises(ValueError, match=r"Duplicate sheet name.+bar\.Nodes"):
        fs.load()


@patch("power_grid_model_io.data_stores.excel_file_store.pd.ExcelWriter")
@patch("power_grid_model_io.data_stores.excel_file_store.pd.DataFrame.to_excel")
def test_save(mock_to_excel: MagicMock, mock_excel_writer: MagicMock):
    # Arrange
    data = TabularData(
        foo=pd.DataFrame([(1, 2.3)], columns=["id", "foo_val"]),
        bar=np.array([(4, 5.6)], dtype=[("id", "i4"), ("bar_val", "f4")]),
    )
    fs = ExcelFileStore(Path("output_data.xlsx"))

    # Act
    fs.save(data=data)

    # Assert
    mock_excel_writer.assert_called_once_with(path=Path("output_data.xlsx"))
    mock_to_excel.assert_any_call(excel_writer=mock_excel_writer().__enter__(), sheet_name="foo")
    mock_to_excel.assert_any_call(excel_writer=mock_excel_writer().__enter__(), sheet_name="bar")


@patch("power_grid_model_io.data_stores.excel_file_store.pd.ExcelWriter")
@patch("power_grid_model_io.data_stores.excel_file_store.pd.DataFrame.to_excel")
def test_save__multiple_files(mock_to_excel: MagicMock, mock_excel_writer):
    # Arrange
    data = TabularData(
        **{
            "nodes": pd.DataFrame(),
            "lines": pd.DataFrame(),
            "foo.colors": pd.DataFrame(),
        }
    )
    fs = ExcelFileStore(Path("output_data.xlsx"), foo=Path("foo.xlsx"), bar=Path("bar.xlsx"))

    output_data_writer = MagicMock()
    foo_writer = MagicMock()
    mock_excel_writer.side_effect = [output_data_writer, foo_writer]

    # Act
    fs.save(data=data)

    # Assert
    mock_excel_writer.assert_any_call(path=Path("output_data.xlsx"))
    mock_to_excel.assert_any_call(excel_writer=output_data_writer.__enter__(), sheet_name="nodes")
    mock_to_excel.assert_any_call(excel_writer=output_data_writer.__enter__(), sheet_name="lines")

    mock_excel_writer.assert_any_call(path=Path("foo.xlsx"))
    mock_to_excel.assert_any_call(excel_writer=foo_writer.__enter__(), sheet_name="colors")


@pytest.mark.parametrize(
    ("column_name", "is_unnamed"),
    [
        param("", False, id="empty"),
        param("id", False, id="id"),
        param("Unnamed: 123_level_1", True, id="unnamed"),
        param("Unnamed", False, id="not_unnamed"),
        param("123", False, id="number_as_str"),
    ],
)
def test_unnamed_pattern(column_name: str, is_unnamed: bool):
    assert bool(ExcelFileStore._unnamed_pattern.fullmatch(column_name)) == is_unnamed


def test_remove_unnamed_column_placeholders():
    # Arrange
    data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["ID", "Unnamed: 123_level_0", "X"])
    store = ExcelFileStore()

    # Act
    result = store._remove_unnamed_column_placeholders(data=data)

    # Assert
    pd.testing.assert_frame_equal(result, pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["ID", "", "X"]))


def test_remove_unnamed_column_placeholders__multi_first():
    # Arrange
    columns = pd.MultiIndex.from_tuples([("ID", "A"), ("Unnamed: 123_level_0", "B"), ("X", "C")])
    data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=columns)
    store = ExcelFileStore()

    # Act
    result = store._remove_unnamed_column_placeholders(data=data)

    # Assert
    columns = pd.MultiIndex.from_tuples([("ID", "A"), ("", "B"), ("X", "C")])
    pd.testing.assert_frame_equal(result, pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=columns))


def test_remove_unnamed_column_placeholders__multi_second():
    # Arrange
    columns = pd.MultiIndex.from_tuples([("ID", ""), ("B", "Unnamed: 123_level_1"), ("C", "kW")])
    data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=columns)
    store = ExcelFileStore()

    # Act
    result = store._remove_unnamed_column_placeholders(data=data)

    # Assert
    columns = pd.MultiIndex.from_tuples([("ID", ""), ("B", ""), ("C", "kW")])
    pd.testing.assert_frame_equal(result, pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=columns))


def test_remove_unnamed_column_placeholders__empty():
    # Arrange
    data = pd.DataFrame()
    store = ExcelFileStore()

    # Act
    result = store._remove_unnamed_column_placeholders(data=data)

    # Assert
    pd.testing.assert_frame_equal(result, data)


@patch("power_grid_model_io.data_stores.excel_file_store.ExcelFileStore._check_duplicate_values")
def test_handle_duplicate_columns(mock_check_duplicate_values: MagicMock):
    # Arrange
    data = pd.DataFrame(
        [  # A    B    C    A    B    A
            # 0    1    2    3    4    5
            [101, 201, 301, 101, 201, 101],
            [102, 202, 302, 111, 202, 102],
            [103, 203, 303, 103, 203, 103],
        ],
        columns=[("A", ""), ("B", ""), ("C", ""), ("A", ""), ("B", ""), ("A", "KW")],
    )
    store = ExcelFileStore()
    mock_check_duplicate_values.return_value = {3: "A_2", 4: "B_2", 5: "A_3"}

    # Act
    with capture_logs() as cap_log:
        actual = store._handle_duplicate_columns(data=data, sheet_name="foo")

    # Assert
    assert len(cap_log) == 3
    assert_log_exists(cap_log, "warning", "Column is renamed", col_name=("A", ""), new_name=("A_2", ""), col_idx=3)
    assert_log_exists(cap_log, "warning", "Column is renamed", col_name=("B", ""), new_name=("B_2", ""), col_idx=4)
    assert_log_exists(cap_log, "warning", "Column is renamed", col_name=("A", "KW"), new_name=("A_3", "KW"), col_idx=5)

    expected = pd.DataFrame(
        [  # A    B    C   A_2  B_2  A_3
            #                         KW
            [101, 201, 301, 101, 201, 101],
            [102, 202, 302, 111, 202, 102],
            [103, 203, 303, 103, 203, 103],
        ],
        columns=[("A", ""), ("B", ""), ("C", ""), ("A_2", ""), ("B_2", ""), ("A_3", "KW")],
    )
    pd.testing.assert_frame_equal(actual, expected)


@patch("power_grid_model_io.data_stores.excel_file_store.ExcelFileStore._check_duplicate_values")
def test_handle_duplicate_columns__multi(mock_check_duplicate_values: MagicMock):
    # Arrange
    data = pd.DataFrame(
        [  # A,1  B,2  C,3  A,1  B,2  A,1
            # 0    1    2    3    4    5
            [101, 201, 301, 101, 201, 101],
            [102, 202, 302, 111, 202, 102],
            [103, 203, 303, 103, 203, 103],
        ],
        columns=pd.MultiIndex.from_tuples([("A", 1), ("B", 2), ("C", 3), ("A", 1), ("B", 2), ("A", 1)]),
    )
    store = ExcelFileStore()
    mock_check_duplicate_values.return_value = {3: ("A_2", 1), 4: ("B_2", 2), 5: ("A_3", 1)}

    # Act
    with capture_logs() as cap_log:
        actual = store._handle_duplicate_columns(data=data, sheet_name="foo")

    # Assert
    assert len(cap_log) == 3
    assert_log_exists(cap_log, "warning", "Column is renamed", col_name=("A", 1), new_name=("A_2", 1), col_idx=3)
    assert_log_exists(cap_log, "warning", "Column is renamed", col_name=("B", 2), new_name=("B_2", 2), col_idx=4)
    assert_log_exists(cap_log, "warning", "Column is renamed", col_name=("A", 1), new_name=("A_3", 1), col_idx=5)

    expected = pd.DataFrame(
        [  # A,1  B,2  C,3 A_2,1 B_2,2 A_3,1
            [101, 201, 301, 101, 201, 101],
            [102, 202, 302, 111, 202, 102],
            [103, 203, 303, 103, 203, 103],
        ],
        columns=pd.MultiIndex.from_tuples([("A", 1), ("B", 2), ("C", 3), ("A_2", 1), ("B_2", 2), ("A_3", 1)]),
    )
    pd.testing.assert_frame_equal(actual, expected)


def test_handle_duplicate_columns__empty():
    # Arrange
    data = pd.DataFrame()
    store = ExcelFileStore()

    # Act
    result = store._handle_duplicate_columns(data=data, sheet_name="foo")

    # Assert
    pd.testing.assert_frame_equal(result, data)


@patch("power_grid_model_io.data_stores.excel_file_store.ExcelFileStore._group_columns_by_index")
def test_check_duplicate_values(mock_group_columns: MagicMock):
    # Arrange
    data = pd.DataFrame(
        [  # A,1  B,2  C,3  A,1  B,2  A,1
            # 0    1    2    3    4    5
            [101, 201, 301, 101, 201, 101],
            [102, 202, 302, 111, 202, 102],
            [103, 203, 303, 103, 203, 103],
        ],
        columns=["A", "B", "C", "A", "B", "A"],
    )
    store = ExcelFileStore()
    mock_group_columns.return_value = {"A": {0, 3, 5}, "B": {1, 4}, "C": {2}}

    # Act
    with capture_logs() as cap_log:
        result = store._check_duplicate_values(data=data, sheet_name="foo")

    # Assert
    assert len(cap_log) == 2
    assert_log_exists(
        cap_log, "error", "Found duplicate column names, with different data", col_name="A", col_idx=[0, 3, 5]
    )
    assert_log_exists(cap_log, "warning", "Found duplicate column names, with same data", col_name="B", col_idx=[1, 4])

    assert result == {3: "A_2", 4: "B_2", 5: "A_3"}


@patch("power_grid_model_io.data_stores.excel_file_store.ExcelFileStore._group_columns_by_index")
def test_check_duplicate_values__multi(mock_group_columns: MagicMock):
    # Arrange
    data = pd.DataFrame(
        [  # A,1  B,2  C,3  A,1  B,2  A,1
            # 0    1    2    3    4    5
            [101, 201, 301, 101, 201, 101],
            [102, 202, 302, 111, 202, 102],
            [103, 203, 303, 103, 203, 103],
        ],
        columns=pd.MultiIndex.from_tuples([("A", 1), ("B", 2), ("C", 3), ("A", 1), ("B", 2), ("A", 1)]),
    )
    store = ExcelFileStore()
    mock_group_columns.return_value = {("A", 1): {0, 3, 5}, ("B", 2): {1, 4}, ("C", 3): {2}}

    # Act
    with capture_logs() as cap_log:
        result = store._check_duplicate_values(data=data, sheet_name="foo")

    # Assert
    assert len(cap_log) == 2
    assert_log_exists(
        cap_log, "error", "Found duplicate column names, with different data", col_name=("A", 1), col_idx=[0, 3, 5]
    )
    assert_log_exists(
        cap_log, "warning", "Found duplicate column names, with same data", col_name=("B", 2), col_idx=[1, 4]
    )

    assert result == {3: ("A_2", 1), 4: ("B_2", 2), 5: ("A_3", 1)}


def test_group_columns_by_index():
    # Arrange
    data = pd.DataFrame(columns=["A", "B", "C", "A", "B", "A"])

    # Act
    grouped = ExcelFileStore._group_columns_by_index(data=data)

    # Assert
    assert grouped == {"A": {0, 3, 5}, "B": {1, 4}, "C": {2}}


def test_group_columns_by_index__multi():
    # Arrange
    data = pd.DataFrame(columns=pd.MultiIndex.from_tuples([("A", 1), ("B", 2), ("C", 3), ("A", 4), ("B", 5), ("A", 6)]))

    # Act
    grouped = ExcelFileStore._group_columns_by_index(data=data)

    # Assert
    assert grouped == {"A": {0, 3, 5}, "B": {1, 4}, "C": {2}}
