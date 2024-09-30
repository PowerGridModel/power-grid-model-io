# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
Vision Excel Converter: Load data from a Vision Excel export file and use a mapping file to convert the data to PGM
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional, Union

from power_grid_model_io.converters.tabular_converter import TabularConverter
from power_grid_model_io.data_stores.base_data_store import LANGUAGE_EN
from power_grid_model_io.data_stores.vision_excel_file_store import VisionExcelFileStore

CONFIG_PATH = Path(__file__).parent.parent / "config" / "excel"
DEFAULT_MAPPING_FILE = CONFIG_PATH / "vision_{language:s}.yaml"


@dataclass
class IdReferenceFields:
    """
    Data class to store langage specific reference fields.
    """

    nodes_table: str
    number: str
    node_number: str
    sub_number: str


class VisionExcelConverter(TabularConverter):
    """
    Vision Excel Converter: Load data from a Vision Excel export file and use a mapping file to convert the data to PGM
    """

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        source_file: Optional[Union[Path, str]] = None,
        language: str = LANGUAGE_EN,
        terms_changed: Optional[dict] = None,
        mapping_file: Optional[Union[Path, str]] = None,
        log_level: int = logging.INFO,
    ):
        _mapping_file = Path(
            mapping_file if mapping_file is not None else str(DEFAULT_MAPPING_FILE).format(language=language)
        )
        if not _mapping_file.exists():
            raise FileNotFoundError(f"No Vision Excel mapping available for language '{language}'")
        self._id_reference: Optional[IdReferenceFields] = None
        source = (
            VisionExcelFileStore(file_path=Path(source_file), language=language, terms_changed=terms_changed)
            if source_file
            else None
        )
        super().__init__(mapping_file=_mapping_file, source=source, log_level=log_level)

    def set_mapping(self, mapping: Mapping[str, Any]) -> None:
        super().set_mapping(mapping)

        if "id_reference" in mapping:
            self._id_reference = IdReferenceFields(**mapping["id_reference"])

    def get_node_id(self, number: int) -> int:
        """
        Get the automatically assigned id of a node
        """
        if self._id_reference is None:
            raise ValueError(f"Missing ID reference definition for {type(self).__name__}.get_node_id()")
        table = self._id_reference.nodes_table
        key = {self._id_reference.number: number}
        return self.get_id(table=table, key=key)

    def get_branch_id(self, table: str, number: int) -> int:
        """
        Get the automatically assigned id of a branch (line, transformer, etc.)
        """
        if self._id_reference is None:
            raise ValueError(f"Missing ID reference definition for {type(self).__name__}.get_branch_id()")
        key = {self._id_reference.number: number}
        return self.get_id(table=table, key=key)

    def get_appliance_id(self, table: str, node_number: int, sub_number: int) -> int:
        """
        Get the automatically assigned id of an appliance (source, load, etc.)
        """
        if self._id_reference is None:
            raise ValueError(f"Missing ID reference definition for {type(self).__name__}.get_appliance_id()")
        key = {self._id_reference.node_number: node_number, self._id_reference.sub_number: sub_number}
        return self.get_id(table=table, key=key)

    def get_virtual_id(self, table: str, obj_name: str, node_number: int, sub_number: int) -> int:
        """
        Get the automatically assigned id of a virtual object (e.g. the internal node of a 'TransformerLoad')
        """
        if self._id_reference is None:
            raise ValueError(f"Missing ID reference definition for {type(self).__name__}.get_virtual_id()")
        key = {self._id_reference.node_number: node_number, self._id_reference.sub_number: sub_number}
        return self.get_id(table=table, name=obj_name, key=key)
