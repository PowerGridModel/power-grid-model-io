# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
PandaPower Converter: Load data in pandapower format and use a mapping file to convert the data to PGM
"""

from pathlib import Path

from power_grid_model_io.converters.tabular_converter import TabularConverter


class PandasConverter(TabularConverter):
    """
    PandaPower Converter: Load data in pandapower format and use a mapping file to convert the data to PGM
    """

    def __init__(self, mapping_file: Path):
        super().__init__(mapping_file=mapping_file)
