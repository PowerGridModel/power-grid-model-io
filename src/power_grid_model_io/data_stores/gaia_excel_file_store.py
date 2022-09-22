# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Gaia Excel file store
"""

from pathlib import Path
from typing import Optional

from power_grid_model_io.data_stores.vision_excel_file_store import VisionExcelFileStore


class GaiaExcelFileStore(VisionExcelFileStore):
    """
    Gaia Excel file store

    Gaia Excel exports are quite simmilar to Vision excel exports. The names of the sheets and columns are quite
    different though, but that will be solved in the mapping file for the TabularConverter. Another difference is
    that Vision exports only one Excel file, where Gaia outputs a separate Excel file containing cable types etc.
    """

    def __init__(self, file_path: Path, types_file: Optional[Path] = None):
        super().__init__(file_path)
        if types_file is not None:
            self._file_paths.append(types_file)
