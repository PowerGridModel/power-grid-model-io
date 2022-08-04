# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from typing import Dict, List, Tuple, Union, cast

from power_grid_model.data_types import ExtraInfo, SingleDataset

from power_grid_model_io.converters.base_converter import BaseConverter
from power_grid_model_io.converters.excel.functions import convert_excel_to_pgm, read_excel_file, read_mapping_file

#           sheet     component attribute  value
Grid = Dict[str, Dict[str, Dict[str, Union[int, float, str, Dict, List]]]]

#            attr      enum       value
Enums = Dict[str, Dict[str, Union[int, float, str]]]

#            unit       value
Units = Dict[str, Union[int, float]]

#            key        data
Mapping = [Dict[str, Union[Grid, Enums, Units]]]


class ExcelConverter(BaseConverter):
    def __init__(self):
        super().__init__()
        self._mapping: Grid = {}
        self._enums: Enums = {}
        self._units: Units = {}

    def set_mapping_file(self, mapping_file: Path) -> None:
        # Read mapping
        if mapping_file.suffix.lower() != ".yaml":
            raise ValueError(f"Mapping file should be a .yaml file, {mapping_file.suffix} provided.")
        self._log.debug("Read mapping file", mapping_file=mapping_file)

        mapping = read_mapping_file(mapping_file)
        self.set_mapping(cast(Grid, mapping.get("grid", {})))
        self.set_enums(cast(Enums, mapping.get("enums", {})))
        self.set_units(cast(Units, mapping.get("units", {})))

    def set_mapping(self, mapping: Grid) -> None:
        self._log.debug(f"Set excel mapping", n_sheets=len(mapping))
        self._mapping = mapping

    def set_enums(self, enums: Enums) -> None:
        self._log.debug(f"Set enum definitions", n_enums=len(enums))
        self._enums = enums

    def set_units(self, units: Units) -> None:
        self._log.debug(f"Set unit definitions", n_units=len(units))
        self._units = units

    def load_input_file(self, src: Path) -> Tuple[SingleDataset, ExtraInfo]:
        # Read input Workbook
        if src.suffix.lower() != ".xlsx":
            raise ValueError(f"Input file should be a .xlsx file, {src.suffix} provided.")
        self._log.debug("Read excel file", src_file=src)
        workbook = read_excel_file(input_file=src, units=self._units, enums=self._enums)

        # Convert Workbook
        input_data, extra_info = convert_excel_to_pgm(workbook=workbook, mapping=self._mapping)
        self._log.debug("Converted excel workbook to power grid model data", src_file=src, n_objects=len(extra_info))

        return input_data, extra_info
