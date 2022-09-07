# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

from power_grid_model_io.data_stores.excel_file_store import ExcelFileStore
from power_grid_model_io.data_types import TabularData


class VisionExcelFileStore(ExcelFileStore):
    def __init__(self, file_path: Path):
        super().__init__(file_path)
        self._header_rows.append(1)  # Units are stored in the row below the column names

    def load(self) -> TabularData:
        data = super().load()
        return data
