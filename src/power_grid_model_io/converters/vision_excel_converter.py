# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
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
        """
        Get the automatically assigned id of a node
        """
        return self.get_id(table="Nodes", key={"Number": number})

    def get_branch_id(self, table: str, number: int) -> int:
        """
        Get the automatically assigned id of a branch (line, transformer, etc.)
        """
        return self.get_id(table=table, key={"Number": number})

    def get_appliance_id(self, table: str, node_number: int, sub_number: int) -> int:
        """
        Get the automatically assigned id of an appliance (source, load, etc.)
        """
        return self.get_id(table=table, key={"Node.Number": node_number, "Subnumber": sub_number})

    def get_virtual_id(self, table: str, obj_name: str, node_number: int, sub_number: int) -> int:
        """
        Get the automatically assigned id of a vitual object (e.g. the internal node of a 'TansformerLoad')
        """
        return self.get_id(table=table, name=obj_name, key={"Node.Number": node_number, "Subnumber": sub_number})
