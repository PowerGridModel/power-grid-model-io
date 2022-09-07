# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import Optional

from power_grid_model_io.data_stores.vision_excel_file_store import VisionExcelFileStore


class GaiaExcelFileStore(VisionExcelFileStore):
    def __init__(self, file_path: Path, types_file: Optional[Path] = None):
        super().__init__(file_path)
        if types_file is not None:
            self._file_paths.append(types_file)
