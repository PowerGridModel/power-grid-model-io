# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import warnings
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest
from structlog.testing import capture_logs

from power_grid_model_io.data_stores.excel_file_store import ExcelFileStore
from power_grid_model_io.data_stores.vision_excel_file_store import VisionExcelFileStore
from power_grid_model_io.data_types.tabular_data import TabularData
from tests.utils import MockExcelFile, assert_log_exists

PandasExcelData = dict[str, pd.DataFrame]


@pytest.fixture
def objects_excel() -> PandasExcelData:
    return {
        "Nodes": pd.DataFrame([["A", 111.0], ["B", 222.0]], columns=("NAME", "U_NOM")),
        "Lines": pd.DataFrame([["C", "A", "B", "X"]], columns=("NAME", "NODE0", "NODE1", "TYPE")),
    }


@pytest.fixture
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


def test_constructor_fragmentation_option_is_not_an_extra_path():
    fs = ExcelFileStore(Path("A.xlsx"), foo=Path("B.xlsx"), suppress_fragmentation_warning=True)

    assert fs.files() == {"": Path("A.xlsx"), "foo": Path("B.xlsx")}


def test_constructor__kwargs():
    # Arrange / Act
    fs = ExcelFileStore(foo=Path("A.xlsx"), bar=Path("B.xls"))

    # Assert
    assert fs.files() == {"foo": Path("A.xlsx"), "bar": Path("B.xls")}


def test_constructor__too_many_args():
    # Too many (> 1) unnamed arguments
    a_path = Path("A.xlsx")
    b_path = Path("B.xls")
    with pytest.raises(TypeError, match=r"1 to 2.*positional arguments.*3.*given"):
        ExcelFileStore(a_path, b_path)  # type: ignore


def test_constructor__invalid_main_file():
    path = Path("A.docx")
    with pytest.raises(ValueError, match=r"Excel.*\.docx"):
        ExcelFileStore(path)


def test_constructor__invalid_named_file():
    a_path = Path("A.xlsx")
    b_path = Path("B.docx")
    with pytest.raises(ValueError, match=r"Extra.*\.docx"):
        ExcelFileStore(a_path, extra=b_path)


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
    with fs.load() as data:
        # Assert
        mock_excel_file.assert_called_once()
        assert mock_excel_file.return_value.is_open
        pd.testing.assert_frame_equal(data["Nodes"], objects_excel["Nodes"])
        pd.testing.assert_frame_equal(data["Lines"], objects_excel["Lines"])
        assert mock_remove_unnamed_column_placeholders.call_args_list[0] == call(data=objects_excel["Nodes"])
        assert mock_remove_unnamed_column_placeholders.call_args_list[1] == call(data=objects_excel["Lines"])
        assert mock_handle_duplicate_columns.call_args_list[0] == call(data=objects_excel["Nodes"], sheet_name="Nodes")
        assert mock_handle_duplicate_columns.call_args_list[1] == call(data=objects_excel["Lines"], sheet_name="Lines")

    assert not mock_excel_file.return_value.is_open


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
        pytest.param("", False, id="empty"),
        pytest.param("id", False, id="id"),
        pytest.param("Unnamed: 123_level_1", True, id="unnamed"),
        pytest.param("Unnamed", False, id="not_unnamed"),
        pytest.param("123", False, id="number_as_str"),
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


def test_process_uuid_columns_warns_by_default_and_can_filter_known_fragmentation():
    data = pd.DataFrame({f"Field{i}GUID": [f"id-{i}"] for i in range(105)})
    store = ExcelFileStore()

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always", pd.errors.PerformanceWarning)
        result = store._process_uuid_columns(data=data, sheet_name="Other")

    assert result is data
    assert any(
        isinstance(item.message, pd.errors.PerformanceWarning)
        and str(item.message).startswith("DataFrame is highly fragmented.")
        for item in emitted
    )
    assert list(result.columns) == [name for i in range(105) for name in (f"Field{i}GUID", f"Field{i}Number")]
    assert result.loc[0, "Field104GUID"] == "id-104"
    assert result.loc[0, "Field0Number"] == 0
    assert result.loc[0, "Field104Number"] == 104

    filtered = ExcelFileStore(suppress_fragmentation_warning=True)
    filtered_data = pd.DataFrame({f"Field{i}GUID": [f"id-{i}"] for i in range(105)})
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always", pd.errors.PerformanceWarning)
        filtered_result = filtered._process_uuid_columns(data=filtered_data, sheet_name="Other")

    assert filtered_result is filtered_data
    assert not any(isinstance(item.message, pd.errors.PerformanceWarning) for item in emitted)
    pd.testing.assert_frame_equal(filtered_result, result)

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        with pytest.raises(pd.errors.PerformanceWarning, match="DataFrame is highly fragmented"):
            ExcelFileStore()._process_uuid_columns(
                pd.DataFrame({f"Field{i}GUID": [f"id-{i}"] for i in range(105)}), "Other"
            )


def test_process_uuid_columns_keeps_existing_number_column_position():
    data = pd.DataFrame({"NodeGUID": ["a", "b"], "Name": ["A", "B"], "NodeNumber": [-1, -1]})

    result = ExcelFileStore()._process_uuid_columns(data=data, sheet_name="Other")

    expected = pd.DataFrame({"NodeGUID": ["a", "b"], "Name": ["A", "B"], "NodeNumber": [0, 1]})
    pd.testing.assert_frame_equal(result, expected)


def test_process_uuid_columns_preserves_two_level_column_labels():
    columns = pd.MultiIndex.from_tuples([("NodeGUID", "id"), ("Name", "text")], names=["field", "unit"])
    data = pd.DataFrame([["a", "A"]], columns=columns)

    result = ExcelFileStore()._process_uuid_columns(data=data, sheet_name="Other")

    expected_columns = pd.MultiIndex.from_tuples(
        [("NodeGUID", "id"), ("NodeNumber", ""), ("Name", "text")], names=["field", "unit"]
    )
    expected = pd.DataFrame([["a", 0, "A"]], columns=expected_columns)
    pd.testing.assert_frame_equal(result, expected)


def test_process_uuid_columns_updates_existing_two_level_number_column():
    columns = pd.MultiIndex.from_tuples([("NodeGUID", "id"), ("NodeNumber", "value")])
    data = pd.DataFrame([["a", -1]], columns=columns)

    result = ExcelFileStore()._process_uuid_columns(data=data, sheet_name="Other")

    expected = pd.DataFrame([["a", 0]], columns=columns)
    pd.testing.assert_frame_equal(result, expected)


def test_process_uuid_columns_uses_special_sheet_subnumber():
    data = pd.DataFrame({"GUID": ["a"], "OtherGUID": ["a"]})

    result = ExcelFileStore()._process_uuid_columns(data=data, sheet_name="Sources")

    assert list(result.columns) == ["GUID", "Subnumber", "OtherGUID", "OtherNumber"]
    assert result.loc[0, "Subnumber"] == result.loc[0, "OtherNumber"] == 0


def test_process_uuid_columns_ignores_non_string_column_labels():
    data = pd.DataFrame([["a", "b"]], columns=[1, "NodeGUID"])

    result = ExcelFileStore()._process_uuid_columns(data=data, sheet_name="Other")

    assert list(result.columns) == [1, "NodeGUID", "NodeNumber"]
    assert result.loc[0, "NodeNumber"] == 0


@pytest.mark.parametrize("levels", [1, 2, 3])
def test_process_uuid_columns_empty_frame_preserves_headers_and_index(levels: int):
    columns = pd.Index(["NodeGUID", "Name"], name="field")
    expected_columns = pd.Index(["NodeGUID", "NodeNumber", "Name"], name="field")
    if levels > 1:
        columns = pd.MultiIndex.from_tuples(
            [("NodeGUID", *("id" for _ in range(levels - 1))), ("Name", *("text" for _ in range(levels - 1)))],
            names=[f"level{i}" for i in range(levels)],
        )
        expected_columns = pd.MultiIndex.from_tuples(
            [columns[0], ("NodeNumber", *("" for _ in range(levels - 1))), columns[1]], names=columns.names
        )
    index = pd.Index([], name="row")
    data = pd.DataFrame(index=index, columns=columns)
    store = ExcelFileStore()

    result = store._process_uuid_columns(data=data, sheet_name="Other")

    pd.testing.assert_frame_equal(result, pd.DataFrame(index=index, columns=expected_columns))
    pd.testing.assert_frame_equal(result[data.columns], data)
    assert store._uuid_cvtr.get_size() == 0


def test_process_uuid_columns_preserves_converter_none_and_nan_semantics():
    data = pd.DataFrame({"NodeGUID": ["a", None, np.nan, "a"]}, index=pd.Index([3, 5, 7, 9], name="row"))
    store = ExcelFileStore()

    result = store._process_uuid_columns(data=data, sheet_name="Other")

    pd.testing.assert_series_equal(result["NodeGUID"], data["NodeGUID"])
    pd.testing.assert_series_equal(
        result["NodeNumber"], pd.Series([0.0, 1.0, np.nan, 0.0], index=data.index, name="NodeNumber")
    )
    assert store._uuid_cvtr.get_keys() == ["a", None]


def test_process_uuid_columns_reuses_converter_across_sheets():
    store = ExcelFileStore()
    nodes = pd.DataFrame({"GUID": ["a", "b"]})
    sources = pd.DataFrame({"GUID": ["b", "c"]})

    first = store._process_uuid_columns(data=nodes, sheet_name="Other")
    second = store._process_uuid_columns(data=sources, sheet_name="Sources")
    again = store._process_uuid_columns(data=nodes, sheet_name="Other")

    assert first["Number"].tolist() == [0, 1]
    assert second["Subnumber"].tolist() == [1, 2]
    pd.testing.assert_frame_equal(first, again)
    assert store._uuid_cvtr.get_keys() == ["a", "b", "c"]


@pytest.mark.parametrize("copy_on_write", [False, True])
def test_process_uuid_columns_preserves_original_in_place_contract(copy_on_write: bool):
    data = pd.DataFrame({"NodeGUID": ["a", "b"], "NodeNumber": [-1, -1], "Name": ["A", "B"]})

    with pd.option_context("mode.copy_on_write", copy_on_write):
        result = ExcelFileStore()._process_uuid_columns(data=data, sheet_name="Other")
        assert result is data
        assert data["NodeNumber"].tolist() == [0, 1]
        result.loc[0, "NodeGUID"] = "changed"
        result.loc[0, "NodeNumber"] = 99
        result.loc[0, "Name"] = "changed"
        assert data.loc[0].tolist() == ["changed", 99, "changed"]


def test_process_uuid_columns_recalculates_positions_after_each_insertion():
    data = pd.DataFrame(
        [["a", "first", "b", -1, "c", "last"]],
        columns=["AGUID", "Name", "BGUID", "BNumber", "CGUID", "Tail"],
    )

    result = ExcelFileStore()._process_uuid_columns(data=data, sheet_name="Other")

    assert result is data
    assert list(result.columns) == ["AGUID", "ANumber", "Name", "BGUID", "BNumber", "CGUID", "CNumber", "Tail"]
    assert result.loc[0].tolist() == ["a", 0, "first", "b", 1, "c", 2, "last"]


def test_fragmentation_option_leaves_other_warnings_visible_and_restores_filters():
    original_insert = pd.DataFrame.insert

    def insert_with_other_warnings(frame, *args, **kwargs):
        warnings.warn("another performance warning", pd.errors.PerformanceWarning, stacklevel=2)
        warnings.warn("a user warning", UserWarning, stacklevel=2)
        return original_insert(frame, *args, **kwargs)

    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always")
        original_filters = warnings.filters.copy()
        with patch.object(pd.DataFrame, "insert", insert_with_other_warnings):
            ExcelFileStore(suppress_fragmentation_warning=True)._process_uuid_columns(
                pd.DataFrame({"NodeGUID": ["a"]}), "Other"
            )
        assert warnings.filters == original_filters

    assert [str(item.message) for item in emitted] == ["another performance warning", "a user warning"]

    with warnings.catch_warnings():
        warnings.simplefilter("error", pd.errors.PerformanceWarning)
        original_filters = warnings.filters.copy()
        with (
            patch.object(pd.DataFrame, "insert", side_effect=RuntimeError("insertion failed")),
            pytest.raises(RuntimeError, match="insertion failed"),
        ):
            ExcelFileStore(suppress_fragmentation_warning=True)._process_uuid_columns(
                pd.DataFrame({"NodeGUID": ["a"]}), "Other"
            )
        assert warnings.filters == original_filters


@pytest.mark.parametrize("suppress", [False, True])
def test_vision_fragmentation_option_applies_when_lazy_sheet_is_consumed(tmp_path: Path, suppress: bool):
    path = tmp_path / "wide-vision.xlsx"
    headers = [f"Field{i}GUID" for i in range(105)]
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame([headers, ["id"] * len(headers), [f"id-{i}" for i in range(105)]]).to_excel(
            writer, sheet_name="Other", index=False, header=False
        )

    store = VisionExcelFileStore(path, suppress_fragmentation_warning=suppress)
    with warnings.catch_warnings(record=True) as emitted:
        warnings.simplefilter("always", pd.errors.PerformanceWarning)
        with store.load() as sheets:
            assert not any(isinstance(item.message, pd.errors.PerformanceWarning) for item in emitted)
            result = sheets["Other"]

    fragmented = [
        item
        for item in emitted
        if isinstance(item.message, pd.errors.PerformanceWarning)
        and str(item.message).startswith("DataFrame is highly fragmented.")
    ]
    assert bool(fragmented) is not suppress
    assert result[("Field104GUID", "id")].iloc[0] == "id-104"
    assert result[("Field104Number", "")].iloc[0] == 104


@pytest.mark.parametrize(
    "columns",
    [
        ["NodeGUID", "NodeGUID"],
        ["NodeGUID", "NodeNumber", "NodeNumber"],
        ["GUIDNumberGUID", "NumberGUIDGUID"],
    ],
)
def test_process_uuid_columns_rejects_ambiguous_labels_before_changing_converter(columns: list[str]):
    data = pd.DataFrame([["a"] * len(columns)], columns=columns)
    before = data.copy(deep=True)
    store = ExcelFileStore()

    with pytest.raises(ValueError, match=r"Ambiguous GUID conversion.*Other"):
        store._process_uuid_columns(data=data, sheet_name="Other")

    pd.testing.assert_frame_equal(data, before)
    assert store._uuid_cvtr.get_size() == 0


@pytest.mark.parametrize("duplicate_guid", ["a", "b"])
def test_process_uuid_columns_after_duplicate_header_preprocessing(duplicate_guid: str):
    columns = pd.MultiIndex.from_tuples(
        [("NodeGUID", "id"), ("NodeGUID", "other"), ("NodeNumber", ""), ("NodeNumber", "other")]
    )
    data = pd.DataFrame([["a", duplicate_guid, -1, -2]], columns=columns)
    store = ExcelFileStore()
    normalized = store._handle_duplicate_columns(data=data, sheet_name="Other")

    result = store._process_uuid_columns(data=normalized, sheet_name="Other")

    expected_columns = pd.MultiIndex.from_tuples(
        [("NodeGUID", "id"), ("NodeGUID_2", "other"), ("NodeNumber", ""), ("NodeNumber_2", "other")]
    )
    pd.testing.assert_frame_equal(result, pd.DataFrame([["a", duplicate_guid, 0, -2]], columns=expected_columns))


def test_vision_sheet_loading_reuses_guid_mapping_and_preserves_original_columns(tmp_path: Path):
    path = tmp_path / "vision.xlsx"
    # Write both header rows explicitly, so pandas does not change the duplicate source headings.
    with pd.ExcelWriter(path) as writer:
        pd.DataFrame(
            [["GUID", "NodeGUID", "NodeGUID", "NodeNumber"], ["id", "id", "other", ""], ["a", "b", "c", -1]]
        ).to_excel(writer, sheet_name="Other", index=False, header=False)
        pd.DataFrame([["GUID", "Name"], ["id", ""], ["b", "source"]]).to_excel(
            writer, sheet_name="Sources", index=False, header=False
        )

    with VisionExcelFileStore(path).load() as sheets:
        nodes = sheets["Other"]
        sources = sheets["Sources"]

        assert nodes[("GUID", "id")].tolist() == ["a"]
        assert nodes[("Number", "")].tolist() == [0]
        assert nodes[("NodeGUID", "id")].tolist() == ["b"]
        assert nodes[("NodeGUID_2", "other")].tolist() == ["c"]
        assert nodes[("NodeNumber", "")].tolist() == [1]
        assert sources[("GUID", "id")].tolist() == ["b"]
        assert sources[("Subnumber", "")].tolist() == [1]


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
