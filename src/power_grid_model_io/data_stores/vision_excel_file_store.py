# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Vision Excel file store
"""
from pathlib import Path

from power_grid_model_io.data_stores.excel_file_store import ExcelFileStore


class VisionExcelFileStore(ExcelFileStore):
    """
    Vision Excel file store

    In Vision files, the second row contains information about the unit of measure.
    Therefore, row 1 (which is row 2 in Excel) is added to the header_rows in the constructor.
    """

    def __init__(self, file_path: Path):
        """
        Args:
            file_path: The main Vision Excel export file
        """
        super().__init__(file_path)
        self._header_rows.append(1)  # Units are stored in the row below the column names
