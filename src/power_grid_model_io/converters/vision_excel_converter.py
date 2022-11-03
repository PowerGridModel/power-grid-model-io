# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Vision Excel Converter: Load data from a Vision Excel export file and use a mapping file to convert the data to PGM
"""

from pathlib import Path
from typing import Optional, Union

from power_grid_model_io.converters.tabular_converter import TabularConverter
from power_grid_model_io.data_stores.vision_excel_file_store import VisionExcelFileStore

DEFAULT_MAPPING_FILE = Path(__file__).parent.parent / "config" / "excel" / "vision_{language:s}.yaml"


class VisionExcelConverter(TabularConverter):
    """
    Vision Excel Converter: Load data from a Vision Excel export file and use a mapping file to convert the data to PGM
    """

    def __init__(self, source_file: Optional[Union[Path, str]] = None, language: str = "en"):
        mapping_file = Path(str(DEFAULT_MAPPING_FILE).format(language=language))
        if not mapping_file.exists():
            raise FileNotFoundError(f"No Vision Excel mapping available for language '{language}'")
        source = VisionExcelFileStore(file_path=Path(source_file)) if source_file else None
        super().__init__(mapping_file=mapping_file, source=source)

    def get_node_id(self, number: int) -> int:
        return self._id_lookup(name="Nodes", key=number)

    def get_branch_id(self, table: str, number: int, sub_number: Optional[int] = None) -> int:
        key = number if sub_number is None else (number, sub_number)
        return self._id_lookup(name=table, key=key)

    def get_appliance_id(self, table: str, node_id: int, sub_number: int) -> int:
        return self._id_lookup(name=table, key=(node_id, sub_number))

    def get_virtual_id(self, table: str, obj_name: str, node_id: int, sub_number: int) -> int:
        return self._id_lookup(name=(table, obj_name), key=(node_id, sub_number))
