# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
import re
from pathlib import Path
from typing import Dict, Hashable, List, Set

import pandas as pd

from power_grid_model_io.data_stores.base_data_store import BaseDataStore
from power_grid_model_io.data_types import TabularData
from power_grid_model_io.utils.modules import assert_dependencies


class ExcelFileStore(BaseDataStore[TabularData]):
    __slots__ = ("_file_paths", "_header_rows")

    def __init__(self, *file_paths: Path):
        super().__init__()
        self._file_paths: List[Path] = list(file_paths)
        self._header_rows: List[int] = [0]
        self._unnamed_pattern = re.compile(r"Unnamed: \d+_level_\d+")

    def load(self) -> TabularData:
        assert_dependencies("excel")
        data: Dict[str, pd.DataFrame] = {}
        for path in self._file_paths:
            if path.suffix.lower() != ".xlsx":
                raise ValueError(f"ExcelFile file should be a .xlsx file, {path.suffix} provided.")
            with path.open(mode="rb") as file_pointer:
                spreadsheet = pd.read_excel(io=file_pointer, sheet_name=None, header=self._header_rows)
            data.update(spreadsheet)

        for sheet_name, sheet_data in data.items():
            self._empty_unnamed_column_indexes(data=sheet_data)
            self._remove_duplicate_columns(sheet_name=sheet_name, data=sheet_data)

        return TabularData(**data)

    def save(self, data: TabularData) -> None:
        assert_dependencies("excel")
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
        match = lambda x: self._unnamed_pattern.fullmatch(str(x))
        columns = (tuple("" if match(idx) else idx for idx in col_idx) for col_idx in data.columns.values)
        data.columns = pd.MultiIndex.from_tuples(columns)

    def _remove_duplicate_columns(self, sheet_name: str, data: pd.DataFrame) -> None:
        if data.empty:
            return

        unique: Dict[Hashable, List[int]] = {}
        to_remove: Set[int] = set()
        to_rename: Dict[int, str] = {}
        columns = data.columns.values
        for i, column in enumerate(columns):
            column = (column,) if not isinstance(column, tuple) else column
            if column in unique:
                same_values = True
                for other_i in unique[column]:
                    same_values = same_values and data.iloc[:, i].equals(data.iloc[:, other_i])
                if same_values:
                    self._log.warning(
                        f"Found duplicate column name, with same data", sheet_name=sheet_name, column=column
                    )
                    to_remove.add(i)
                else:
                    self._log.error(
                        f"Found duplicate column name, with different data", sheet_name=sheet_name, column=column
                    )
                for idx, other_i in enumerate(unique[column][1:] + [i]):
                    to_remove -= {other_i}
                    to_rename[other_i] = (f"{column[0]}_{idx + 1}",) + column[1:]
            else:
                unique[column] = []
            unique[column].append(i)

        for i in to_rename:
            self._log.warning(f"Column is renamed", sheet_name=sheet_name, col_name=columns[i], new_name=to_rename[i])
            columns[i] = to_rename[i]
        data.columns = pd.MultiIndex.from_tuples(columns)

        for i in to_remove:
            self._log.debug(f"Column is removed", sheet_name=sheet_name, col_name=columns[i])
        data.drop(data.iloc[:, list(to_remove)], axis=1, inplace=True)
