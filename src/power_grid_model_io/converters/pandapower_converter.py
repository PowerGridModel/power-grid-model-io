# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
PandaPower Converter: Load data in pandapower format and use a mapping file to convert the data to PGM
"""

from pathlib import Path
from typing import Dict, Optional, Union

import pandas as pd

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

    def _id_lookup(self, component: str, row: pd.Series) -> int:

        row_dict = {}
        for key, value in row.to_dict().items():
            if key == "bus" or key.endswith("_bus"):
                key = "index"
            row_dict[key] = value

        items = sorted(row_dict.items(), key=lambda x: str(x[0]))
        key = component + ":" + ",".join(f"{k}={v}" for k, v in items)
        return self._lookup(item={"component": component, "row": row_dict}, key=key)
