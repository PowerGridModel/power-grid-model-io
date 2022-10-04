# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Excel File Store
"""

import re
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd

from power_grid_model_io.data_stores.base_data_store import BaseDataStore
from power_grid_model_io.data_types import TabularData


class ExcelFileStore(BaseDataStore[TabularData]):
    """
    Excel File Store

    The first row of each sheet is expected to contain the column names, unless specified differently by an extension
    of this class. Columns with duplicate names (on the same sheet) are either removed (if they contain exactly the
    same values) or renamed.
    """

    __slots__ = ("_file_paths", "_header_rows")

    _unnamed_pattern: re.Pattern = re.compile(r"Unnamed: \d+_level_\d+")

    def __init__(self, *file_paths: Path):
        super().__init__()
        self._file_paths: List[Path] = list(file_paths)
        self._header_rows: List[int] = [0]

    def load(self) -> TabularData:
        """
        Load one or more Excel file as tabular data.
        """
        data: Dict[str, pd.DataFrame] = {}
        for path in self._file_paths:
            if path.suffix.lower() != ".xlsx":
                raise ValueError(f"ExcelFile file should be a .xlsx file, {path.suffix} provided.")
            with path.open(mode="rb") as file_pointer:
                spreadsheet = pd.read_excel(io=file_pointer, sheet_name=None, header=self._header_rows)
            data.update(spreadsheet)

        for sheet_name, sheet_data in data.items():
            self._empty_unnamed_column_indexes(data=sheet_data)
            data[sheet_name] = self._handle_duplicate_columns(sheet_name=sheet_name, data=sheet_data)

        return TabularData(**data)

    def save(self, data: TabularData) -> None:
        """
        TODO: Test and finalize this method
        """
        if len(self._file_paths) != 1:
            raise ValueError(f"ExcelFileStore can only write to a single .xlsx file, {len(self._file_paths)} provided.")
        with self._file_paths[0].open(mode="wb") as file_pointer:
            for sheet_name, sheet_data in data.items():
                sheet_data.to_excel(
                    excel_writer=file_pointer,
                    sheet_name=sheet_name,
                )

    def _empty_unnamed_column_indexes(self, data: pd.DataFrame) -> None:
        if data.empty:
            return

        def is_unnamed(col_name):
            return self._unnamed_pattern.fullmatch(str(col_name))

        columns = (tuple("" if is_unnamed(idx) else idx for idx in col_idx) for col_idx in data.columns.values)
        data.columns = pd.MultiIndex.from_tuples(columns)

    def _handle_duplicate_columns(self, sheet_name: str, data: pd.DataFrame) -> pd.DataFrame:
        if data.empty:
            return data

        grouped = self._group_columns_by_index(data=data)
        to_remove, to_rename = self._check_duplicate_values(sheet_name=sheet_name, data=data, grouped=grouped)

        columns = data.columns.values
        if to_rename:
            for col_idx, new_name in to_rename.items():
                self._log.warning(
                    "Column is renamed",
                    sheet_name=sheet_name,
                    col_name=columns[col_idx],
                    new_name=new_name,
                    col_idx=col_idx,
                )
                columns[col_idx] = new_name
            data.columns = pd.MultiIndex.from_tuples(columns)

        for col_idx in to_remove:
            self._log.debug("Column is removed", sheet_name=sheet_name, col_name=columns[col_idx], col_idx=col_idx)
        all_columns = set(range(len(data.columns)))
        to_keep = all_columns - to_remove
        return data.iloc[:, sorted(to_keep)]

    def _group_columns_by_index(self, data: pd.DataFrame) -> Dict[Tuple[str, ...], Set[int]]:
        grouped: Dict[Tuple[str, ...], Set[int]] = {}
        columns = data.columns.values
        for col_idx, col_name in enumerate(columns):
            col_name = (col_name,) if not isinstance(col_name, tuple) else col_name
            if col_name not in grouped:
                grouped[col_name] = set()
            grouped[col_name].add(col_idx)
        return grouped

    def _check_duplicate_values(
        self, sheet_name: str, data: pd.DataFrame, grouped: Dict[Tuple[str, ...], Set[int]]
    ) -> Tuple[Set[int], Dict[int, Tuple[str, ...]]]:

        to_remove: Set[int] = set()
        to_rename: Dict[int, Tuple[str, ...]] = {}

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
                    column=col_name,
                    col_idx=sorted(col_idxs),
                )
                to_remove |= dup_idxs
            else:
                self._log.error(
                    "Found duplicate column names, with different data",
                    sheet_name=sheet_name,
                    column=col_name,
                    col_idx=sorted(col_idxs),
                )
                for counter, dup_idx in enumerate(sorted(dup_idxs), start=2):
                    to_rename[dup_idx] = (f"{col_name[0]}_{counter}",) + col_name[1:]

        return to_remove, to_rename
