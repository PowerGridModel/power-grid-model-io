# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
Excel File Store
"""

import re
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import pandas as pd

from power_grid_model_io.data_stores.base_data_store import (
    DICT_KEY_NUMBER,
    DICT_KEY_SUBNUMBER,
    VISION_EXCEL_LAN_DICT,
    BaseDataStore,
)
from power_grid_model_io.data_types import LazyDataFrame, TabularData
from power_grid_model_io.utils.uuid_excel_cvtr import (
    UUID2IntCvtr,
    add_guid_values_to_cvtr,
    get_special_key_map,
    special_nodes_en,
    special_nodes_nl,
    update_column_names,
)


class ExcelFileStore(BaseDataStore[TabularData]):
    """
    Excel File Store

    The first row of each sheet is expected to contain the column names, unless specified differently by an extension
    of this class. Columns with duplicate names (on the same sheet) are either removed (if they contain exactly the
    same values) or renamed.
    """

    __slots__ = ("_file_paths", "_excel_files", "_header_rows")

    _unnamed_pattern: re.Pattern = re.compile(r"Unnamed: \d+_level_\d+")

    def __init__(
        self,
        file_path: Optional[Path] = None,
        *,
        language: str = "en",
        terms_changed: Optional[dict] = None,
        **extra_paths: Path,
    ):
        super().__init__()
        # Create a dictionary of all supplied file paths:
        # {"": file_path, extra_name[0]: extra_path[0], extra_name[1]: extra_path[1], ...}
        self._file_paths: Dict[str, Path] = {}
        if file_path is not None:
            self._file_paths[""] = file_path
        for name, path in extra_paths.items():
            self._file_paths[name] = path

        for name, path in self._file_paths.items():
            if path.suffix.lower() not in {".xls", ".xlsx"}:
                name = name.title() if name else "Excel"
                raise ValueError(f"{name} file should be a .xls or .xlsx file, {path.suffix} provided.")

        self._header_rows: List[int] = [0]
        self._language = language
        self._vision_excel_key_mapping = VISION_EXCEL_LAN_DICT[self._language]
        self._terms_changed = terms_changed if terms_changed is not None else {}
        self._uuid_cvtr = UUID2IntCvtr()

    def files(self) -> Dict[str, Path]:
        """
        The files as supplied in the constructor. Note that the file names are read-only.

        Returns: A copy of the file paths as set in the constructor.
        """
        return self._file_paths.copy()

    def load(self) -> TabularData:
        """
        Load one or more Excel file as tabular data.

        Returns: The contents of all the Excel file supplied in the constructor. The tables of the main file will
        have no prefix, while the tables of all the extra files will be prefixed with the name of the key word argument
        as supplied in the constructor.
        """

        def lazy_sheet_loader(xls_file: pd.ExcelFile, xls_sheet_name: str):
            def sheet_loader():
                sheet_data = xls_file.parse(xls_sheet_name, header=self._header_rows)
                sheet_data = self._remove_unnamed_column_placeholders(data=sheet_data)
                sheet_data = self._handle_duplicate_columns(data=sheet_data, sheet_name=xls_sheet_name)
                sheet_data = self._process_uuid_columns(data=sheet_data, sheet_name=xls_sheet_name)
                sheet_data = self._update_column_names(data=sheet_data)
                return sheet_data

            return sheet_loader

        data: Dict[str, LazyDataFrame] = {}
        for name, path in self._file_paths.items():
            excel_file = pd.ExcelFile(path)
            for sheet_name in excel_file.sheet_names:
                loader = lazy_sheet_loader(excel_file, sheet_name)
                if name != "":  # If the Excel file is not the main file, prefix the sheet name with the file name
                    sheet_name = f"{name}.{sheet_name}"
                if sheet_name in data:
                    raise ValueError(f"Duplicate sheet name '{sheet_name}'")
                data[sheet_name] = loader
        return TabularData(**data)

    def save(self, data: TabularData) -> None:
        """
        Store tabular data as one or more Excel file.

        Args:
            data: Tha data to store. The keys of the tables will be the names of the spread sheets in the excel
            files. Table names with a prefix corresponding to the name of the key word argument as supplied in the
            constructor will be stored in the associated files.
        """

        # First group all sheets per file. Each sheet name that starts with a file name is assigned to that file.
        # Sheets that don't start with a file name are assigned to the first file.
        sheets: Dict[str, Dict[str, pd.DataFrame]] = {file_name: {} for file_name in self._file_paths}
        for sheet_name, sheet_data in data.items():
            if "." in sheet_name:
                file_name, alt_sheet_name = sheet_name.split(".", maxsplit=1)
                if file_name in self._file_paths:
                    sheets[file_name][alt_sheet_name] = sheet_data
                    continue
            sheets[""][sheet_name] = sheet_data

        # Create an Excel file if there is at least one sheet.
        for file_name, file_path in self._file_paths.items():
            if not sheets[file_name]:
                continue
            with pd.ExcelWriter(path=file_path) as excel_writer:  # pylint: disable=abstract-class-instantiated
                for sheet_name, sheet_data in sheets[file_name].items():
                    if not isinstance(sheet_data, pd.DataFrame):
                        sheet_data = pd.DataFrame(sheet_data)
                    sheet_data.to_excel(
                        excel_writer=excel_writer,
                        sheet_name=sheet_name,
                    )

    def _remove_unnamed_column_placeholders(self, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data

        def is_unnamed(col_name) -> bool:
            return bool(self._unnamed_pattern.fullmatch(str(col_name)))

        if data.columns.nlevels == 1:
            data.columns = pd.Index("" if is_unnamed(idx) else idx for idx in data.columns.values)
        else:
            data.columns = pd.MultiIndex.from_tuples(
                [tuple("" if is_unnamed(idx) else idx for idx in col_idx) for col_idx in data.columns.values]
            )

        return data

    def _handle_duplicate_columns(self, data: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        if data.empty:
            return data

        to_rename = self._check_duplicate_values(sheet_name=sheet_name, data=data)
        if to_rename:
            columns = data.columns.values.copy()
            for col_idx, new_name in to_rename.items():
                new_name = new_name[0] if isinstance(new_name, tuple) else new_name
                full_new_name = (new_name, columns[col_idx][1])
                self._log.warning(
                    "Column is renamed",
                    sheet_name=sheet_name,
                    col_name=columns[col_idx],
                    new_name=full_new_name,
                    col_idx=col_idx,
                )
                columns[col_idx] = full_new_name

            if data.columns.nlevels == 1:
                data.columns = pd.Index(columns)
            else:
                data.columns = pd.MultiIndex.from_tuples(columns)

        return data

    def _check_duplicate_values(self, sheet_name: str, data: pd.DataFrame) -> Dict[int, Union[str, Tuple[str, ...]]]:
        grouped = self._group_columns_by_index(data=data)

        to_rename: Dict[int, Union[str, Tuple[str, ...]]] = {}

        for col_name, col_idxs in grouped.items():
            # No duplicate column names
            if len(col_idxs) == 1:
                continue

            # Select the first column as a reference
            ref_idx = min(col_idxs)

            # Select the rest as duplicates
            dup_idxs = col_idxs - {ref_idx}

            same_values = all(data.iloc[:, dup_idx].equals(data.iloc[:, ref_idx]) for dup_idx in dup_idxs)
            if same_values:
                self._log.warning(
                    "Found duplicate column names, with same data",
                    sheet_name=sheet_name,
                    col_name=col_name,
                    col_idx=sorted(col_idxs),
                )
            else:
                self._log.error(
                    "Found duplicate column names, with different data",
                    sheet_name=sheet_name,
                    col_name=col_name,
                    col_idx=sorted(col_idxs),
                )
            for counter, dup_idx in enumerate(sorted(dup_idxs), start=2):
                if isinstance(col_name, tuple):
                    to_rename[dup_idx] = (f"{col_name[0]}_{counter}",) + col_name[1:]
                else:
                    to_rename[dup_idx] = f"{col_name}_{counter}"

        return to_rename

    def _process_uuid_columns(self, data: pd.DataFrame, sheet_name: str) -> pd.DataFrame:
        first_level = data.columns.get_level_values(0)
        guid_columns = first_level[first_level.str.endswith("GUID")]

        sheet_key_mapping = get_special_key_map(
            sheet_name=sheet_name, nodes_en=special_nodes_en, nodes_nl=special_nodes_nl
        )

        for guid_column in guid_columns:
            nr = VISION_EXCEL_LAN_DICT[self._language][DICT_KEY_NUMBER]
            add_guid_values_to_cvtr(data, guid_column, self._uuid_cvtr)
            new_column_name = guid_column.replace("GUID", nr)
            if guid_column == "GUID" and sheet_key_mapping not in (None, {}):
                new_column_name = guid_column.replace("GUID", sheet_key_mapping[DICT_KEY_SUBNUMBER])
            guid_column_pos = first_level.tolist().index(guid_column)
            try:
                data.insert(guid_column_pos + 1, new_column_name, data[guid_column].apply(self._uuid_cvtr.query))
            except ValueError:
                data[new_column_name] = data[guid_column].apply(self._uuid_cvtr.query)

        return data

    def _update_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        update_column_names(data, self._terms_changed)
        return data

    @staticmethod
    def _group_columns_by_index(data: pd.DataFrame) -> Dict[Union[str, Tuple[str, ...]], Set[int]]:
        grouped: Dict[Union[str, Tuple[str, ...]], Set[int]] = {}
        columns = data.columns.values
        for col_idx, col_name in enumerate(columns):
            if col_name[0] not in grouped:
                grouped[col_name[0]] = set()
            grouped[col_name[0]].add(col_idx)
        return grouped
