# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Gaia Excel Converter: Load data from a Gaia Excel export file and use a mapping file to convert the data to PGM
"""

from pathlib import Path
from typing import Optional

from power_grid_model_io.converters.tabular_converter import TabularConverter
from power_grid_model_io.data_stores.gaia_excel_file_store import GaiaExcelFileStore

DEFAULT_MAPPING_FILE = Path(__file__).parent.parent / "config" / "excel" / "gaia_{language:s}.yaml"


class GaiaExcelConverter(TabularConverter):
    """
    Gaia Excel Converter: Load data from a Gaia Excel export file and use a mapping file to convert the data to PGM
    """

    def __init__(self, source_file: Optional[Path] = None, types_file: Optional[Path] = None, language: str = "en"):
        mapping_file = Path(str(DEFAULT_MAPPING_FILE).format(language=language))
        if not mapping_file.exists():
            raise FileNotFoundError(f"No Gaia Excel mapping available for language '{language}'")
        source = GaiaExcelFileStore(file_path=source_file, types_file=types_file) if source_file else None
        super().__init__(mapping_file=mapping_file, source=source)
