# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
PandaPower Converter: Load data in pandapower format and use a mapping file to convert the data to PGM
"""

from pathlib import Path
from typing import Dict, Optional, Union

from power_grid_model_io.converters.tabular_converter import TabularConverter

DEFAULT_MAPPING_FILE = Path(__file__).parent.parent / "config" / "pandas" / "pandapower.yaml"


class PandaPowerConverter(TabularConverter):
    """
    PandaPower Converter: Load data in pandapower format and use a mapping file to convert the data to PGM
    """

    def __init__(
        self,
        std_types: Optional[Dict[str, Dict[str, Dict[str, Union[float, int, str]]]]] = None,
        mapping_file: Path = DEFAULT_MAPPING_FILE,
    ):
        super().__init__(mapping_file=mapping_file)
        self._std_types = std_types

    def get_trafo_vector_group(self, std_type: str) -> str:
        """
        Get the vector group from the std_type table
        """
        if self._std_types is not None:
            trafo = self._std_types.get("trafo", {})
            if std_type in trafo:
                return str(trafo[std_type]["vector_group"])
        return std_type

    def get_trafo3w_vector_group(self, std_type: str) -> str:
        """
        Get the vector of a three winding transformer group from the std_type table
        """
        if self._std_types is not None:
            trafo3w = self._std_types.get("trafo3w", {})
            if std_type in trafo3w:
                return str(trafo3w[std_type]["vector_group"])
        return std_type
