# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
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

from ...utils import assert_log_exists, assert_log_match

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


@patch("power_grid_model_io.data_stores.excel_file_store.pd.ExcelWriter")
@patch("power_grid_model_io.data_stores.excel_file_store.pd.DataFrame.to_excel")
def test_excel_file_store__save(mock_to_excel: MagicMock, mock_excel_writer: MagicMock):
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
def test_excel_file_store__save__multiple_files(mock_to_excel: MagicMock, mock_excel_writer):
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
    with capture_logs() as cap_log:
        result = store._remove_unnamed_column_placeholders(data=data, sheet_name="foo")

    # Assert
    assert len(cap_log) == 1
    assert_log_match(cap_log[0], "warning", "Column is renamed", col_name="Unnamed: 123_level_0")
    pd.testing.assert_frame_equal(result, pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=["ID", "", "X"]))


def test_remove_unnamed_column_placeholders__multi_first():
    # Arrange
    columns = pd.MultiIndex.from_tuples([("ID", "A"), ("Unnamed: 123_level_0", "B"), ("X", "C")])
    data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=columns)
    store = ExcelFileStore()

    # Act
    with capture_logs() as cap_log:
        result = store._remove_unnamed_column_placeholders(data=data, sheet_name="foo")

    # Assert
    assert len(cap_log) == 1
    assert_log_match(cap_log[0], "warning", "Column is renamed", col_name="Unnamed: 123_level_0")
    columns = pd.MultiIndex.from_tuples([("ID", "A"), ("", "B"), ("X", "C")])
    pd.testing.assert_frame_equal(result, pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=columns))


def test_remove_unnamed_column_placeholders__multi_seccond():
    # Arrange
    columns = pd.MultiIndex.from_tuples([("ID", ""), ("B", "Unnamed: 123_level_1"), ("C", "kW")])
    data = pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=columns)
    store = ExcelFileStore()

    # Act
    with capture_logs() as cap_log:
        result = store._remove_unnamed_column_placeholders(data=data, sheet_name="foo")

    # Assert
    assert len(cap_log) == 1
    assert_log_match(cap_log[0], "warning", "Column is renamed", col_name="Unnamed: 123_level_1")
    columns = pd.MultiIndex.from_tuples([("ID", ""), ("B", ""), ("C", "kW")])
    pd.testing.assert_frame_equal(result, pd.DataFrame([[1, 2, 3], [4, 5, 6]], columns=columns))


@patch("power_grid_model_io.data_stores.excel_file_store.ExcelFileStore._check_duplicate_values")
def test_handle_duplicate_columns(mock_check_duplicate_values: MagicMock):
    # Arrange
    data = pd.DataFrame(
        [  #  A    B    C    A    B    A    B
            # 0    1    2    3    4    5    6
            [101, 201, 301, 101, 201, 101, 201],
            [102, 202, 302, 111, 202, 102, 202],
            [103, 203, 303, 103, 203, 103, 203],
        ],
        columns=["A", "B", "C", "A", "B", "A", "B"],
    )
    store = ExcelFileStore()
    mock_check_duplicate_values.return_value = ({4, 5, 6}, {3: "A_2"})

    # Act
    with capture_logs() as cap_log:
        actual = store._handle_duplicate_columns(data=data, sheet_name="foo")

    # Assert
    assert len(cap_log) == 4
    assert_log_exists(cap_log, "warning", "Column is renamed", col_name="A", new_name="A_2", col_idx=3)
    assert_log_exists(cap_log, "debug", "Column is removed", col_name="B", col_idx=4)
    assert_log_exists(cap_log, "debug", "Column is removed", col_name="A", col_idx=5)
    assert_log_exists(cap_log, "debug", "Column is removed", col_name="B", col_idx=6)

    expected = pd.DataFrame(
        [  #  A    B    C   A_2
            [101, 201, 301, 101],
            [102, 202, 302, 111],
            [103, 203, 303, 103],
        ],
        columns=["A", "B", "C", "A_2"],
    )
    pd.testing.assert_frame_equal(actual, expected)


@patch("power_grid_model_io.data_stores.excel_file_store.ExcelFileStore._check_duplicate_values")
def test_handle_duplicate_columns__multi(mock_check_duplicate_values: MagicMock):
    # Arrange
    data = pd.DataFrame(
        [  # A,1  B,2  C,3  A,1  B,2  A,1  B,2
            # 0    1    2    3    4    5    6
            [101, 201, 301, 101, 201, 101, 201],
            [102, 202, 302, 111, 202, 102, 202],
            [103, 203, 303, 103, 203, 103, 203],
        ],
        columns=pd.MultiIndex.from_tuples([("A", 1), ("B", 2), ("C", 3), ("A", 1), ("B", 2), ("A", 1), ("B", 2)]),
    )
    store = ExcelFileStore()
    mock_check_duplicate_values.return_value = ({4, 5, 6}, {3: ("A_2", 1)})

    # Act
    with capture_logs() as cap_log:
        actual = store._handle_duplicate_columns(data=data, sheet_name="foo")

    # Assert
    assert len(cap_log) == 4
    assert_log_exists(cap_log, "warning", "Column is renamed", col_name=("A", 1), new_name=("A_2", 1), col_idx=3)
    assert_log_exists(cap_log, "debug", "Column is removed", col_name=("B", 2), col_idx=4)
    assert_log_exists(cap_log, "debug", "Column is removed", col_name=("A", 1), col_idx=5)
    assert_log_exists(cap_log, "debug", "Column is removed", col_name=("B", 2), col_idx=6)

    expected = pd.DataFrame(
        [  # A,1  B,2  C,3 A_2,1
            [101, 201, 301, 101],
            [102, 202, 302, 111],
            [103, 203, 303, 103],
        ],
        columns=pd.MultiIndex.from_tuples([("A", 1), ("B", 2), ("C", 3), ("A_2", 1)]),
    )
    pd.testing.assert_frame_equal(actual, expected)
