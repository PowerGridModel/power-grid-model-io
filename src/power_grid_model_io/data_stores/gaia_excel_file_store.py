# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Gaia Excel file store
"""

from pathlib import Path
from typing import Optional

from power_grid_model_io.data_stores.excel_file_store import ExcelFileStore


class GaiaExcelFileStore(ExcelFileStore):
    """
    Gaia Excel file store

    Gaia Excel exports are quite similar to Vision Excel exports. The names of the sheets and columns are quite
    different though, but that will be solved in the mapping file for the TabularConverter. Another difference is
    that Vision exports only one Excel file, where Gaia outputs a separate Excel file containing cable types etc.
    """

    def __init__(self, file_path: Path, types_file: Optional[Path] = None):
        """
        Args:
            file_path: The main Gaia Excel export file
            types_file: The Excel file storing cable types etc.
        """
        if types_file is None:
            super().__init__(file_path)
        else:
            super().__init__(file_path, types=types_file)
        self._header_rows.append(1)  # Units are stored in the row below the column names
