# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
Vision Excel file store
"""
from pathlib import Path
from typing import Optional

from power_grid_model_io.data_stores.base_data_store import LANGUAGE_EN
from power_grid_model_io.data_stores.excel_file_store import ExcelFileStore


class VisionExcelFileStore(ExcelFileStore):
    """
    Vision Excel file store

    In Vision files, the second row contains information about the unit of measure.
    Therefore, row 1 (which is row 2 in Excel) is added to the header_rows in the constructor.
    """

    def __init__(
        self,
        file_path: Path,
        language: str = LANGUAGE_EN,
        terms_changed: Optional[dict] = None,
    ):
        """
        Args:
            file_path: The main Vision Excel export file
        """
        super().__init__(file_path, language=language, terms_changed=terms_changed)
        self._header_rows.append(1)  # Units are stored in the row below the column names
