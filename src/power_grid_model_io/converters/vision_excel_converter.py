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
DEFAULT_MAPPING_KEYS = {
    # Table names
    "Nodes": {
        "en": "Nodes",
        "nl": "Knooppunten",
    },
    "Cables": {
        "en": "Cables",
        "nl": "Kabels",
    },
    "Lines": {
        "en": "Lines",
        "nl": "Verbindingen",
    },
    "Links": { # needs to be in translation table, even though translated value is the same
        "en": "Links",
        "nl": "Links",
    },
    "Reactance coils": {
        "en": "Reactance coils",
        "nl": "Smoorspoelen",
    },
    "Transformers": {
        "en": "Transformers",
        "nl": "Transformatoren",
    },
    "Special transformers": {
        "en": "Special transformers",
        "nl": "Speciale transformatoren",
    },
    "Transformer loads": {
        "en": "Transformer loads",
        "nl": "Transformatorbelastingen",
    },
    "Sources": {
        "en": "Sources",
        "nl": "Netvoedingen",
    },
    "Synchronous generators": {
        "en": "Synchronous generators",
        "nl": "Synchrone generatoren",
    },
    "Wind turbines": {
        "en": "Wind turbines",
        "nl": "Windturbines",
    },
    "Loads": {
        "en": "Loads",
        "nl": "Belastingen",
    },
    "Zigzag transformers": {
        "en": "Zigzag transformers",
        "nl": "Nulpuntstransformatoren",
    },
    "Capacitors": {
        "en": "Capacitors",
        "nl": "Condensatoren",
    },
    "Reactors": {
        "en": "Reactors",
        "nl": "Spoelen",
    },
    "Pvs": {
        "en": "Pvs",
        "nl": "Pv's",
    },
    "Three winding transformers": {
        "en": "Three winding transformers",
        "nl": "Driewikkelingstransformatoren",
    },

    # (Sub) Field names
    "Number": {
        "en": "Number",
        "nl": "Nummer",
    },
    "Node.Number": {
        "en": "Node.Number",
        "nl": "Knooppunt.Nummer",
    },
    "From.Number": {
        "en": "From.Number",
        "nl": "Van.Nummer",
    },
    "To.Number": {
        "en": "To.Number",
        "nl": "Naar.Nummer",
    },
    "Subnumber": {
        "en": "Subnumber",
        "nl": "Subnummer",
    },

    "transformer": {
        "en": "transformer",
        "nl": "transformer",
    },
    "internal_node": {
        "en": "internal_node",
        "nl": "internal_node",
    },
    "load": {
        "en": "load",
        "nl": "load",
    },
    "generation": {
        "en": "generation",
        "nl": "generation",
    },
    "pv_generation": {
        "en": "pv_generation",
        "nl": "pv_generation",
    },

    # For consideration:
    #   from_status: From.Switch state, to_status: To.Switch state
    #   i_0, p_0, conn_str, neutral_grounding, tapside, tap*
    #   (r|x)_grounding_(from|to)

}
# keyword -> DEFAULT_MAPPING_KEYS[keyword][language]


class VisionExcelConverter(TabularConverter):
    """
    Vision Excel Converter: Load data from a Vision Excel export file and use a mapping file to convert the data to PGM
    """

    def __init__(self, source_file: Optional[Union[Path, str]] = None, language: str = "en"):
        mapping_file = Path(str(DEFAULT_MAPPING_FILE).format(language=str(language)))
        if not mapping_file.exists():
            raise FileNotFoundError(f"No Vision Excel mapping available for language '{str(language)}'")
        source = VisionExcelFileStore(file_path=Path(source_file)) if source_file else None
        self.language = str(language)
        super().__init__(mapping_file=mapping_file, source=source)

    def get_node_id(self, number: int) -> int:
        """
        Get the automatically assigned id of a node
        If the language is set to e.g. "nl", Nodes.Number is mapped to Knooppunten.Nummer
        """
        language = self.language or "en"
        return self.get_id(
            table=DEFAULT_MAPPING_KEYS["Nodes"][language],
            key={DEFAULT_MAPPING_KEYS["Number"][language]: number}
        )

    def get_branch_id(self, table: str, number: int) -> int:
        """
        Get the automatically assigned id of a branch (line, transformer, etc.)
        """
        language = self.language or "en"
        return self.get_id(
            table=DEFAULT_MAPPING_KEYS[table][language],
            key={DEFAULT_MAPPING_KEYS["Number"][language]: number}
        )

    def get_appliance_id(self, table: str, node_number: int, sub_number: int) -> int:
        """
        Get the automatically assigned id of an appliance (source, load, etc.)
        """
        language = self.language or "en"
        return self.get_id(
            table=DEFAULT_MAPPING_KEYS[table][language],
            key={
                DEFAULT_MAPPING_KEYS["Node.Number"][language]: node_number,
                DEFAULT_MAPPING_KEYS["Subnumber"][language]: sub_number,
            }
        )

    def get_virtual_id(self, table: str, obj_name: str, node_number: int, sub_number: int) -> int:
        """
        Get the automatically assigned id of a virtual object (e.g. the internal node of a 'TransformerLoad')
        """
        language = self.language or "en"
        return self.get_id(
            table=DEFAULT_MAPPING_KEYS[table][language],
            name=DEFAULT_MAPPING_KEYS[obj_name][language],
            key={
                DEFAULT_MAPPING_KEYS["Node.Number"][language]: node_number,
                DEFAULT_MAPPING_KEYS["Subnumber"][language]: sub_number,
            }
        )
