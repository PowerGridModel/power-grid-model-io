# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from typing import Dict, Literal, Optional, Union, cast

from power_grid_model.data_types import Dataset

from power_grid_model_io.converters.base_converter import BaseConverter
from power_grid_model_io.converters.excel.functions import convert_excel_to_pgm
from power_grid_model_io.data_types import ExtraInfoLookup, TabularData
from power_grid_model_io.mappings.tabular_mapping import Tables, TabularMapping
from power_grid_model_io.mappings.unit_mapping import UnitMapping, Units
from power_grid_model_io.mappings.value_mapping import ValueMapping, Values
from power_grid_model_io.utils.modules import assert_dependencies

assert_dependencies("tabular")

import yaml


class TabularConverter(BaseConverter[TabularData]):
    def __init__(self, mapping_file: Optional[Path] = None):
        super().__init__()
        self._mapping: TabularMapping = TabularMapping(mapping={})
        self._units: Optional[UnitMapping] = None
        self._substitution: Optional[ValueMapping] = None
        if mapping_file is not None:
            self.set_mapping_file(mapping_file=mapping_file)

    def set_mapping_file(self, mapping_file: Path) -> None:
        # Read mapping
        if mapping_file.suffix.lower() != ".yaml":
            raise ValueError(f"Mapping file should be a .yaml file, {mapping_file.suffix} provided.")
        self._log.debug("Read mapping file", mapping_file=mapping_file)

        with open(mapping_file, "r", encoding="utf-8") as mapping_stream:
            mapping: Dict[Literal["grid", "units", "substitutions"], Union[Tables, Units, Values]] = yaml.safe_load(
                mapping_stream
            )
        if "grid" not in mapping:
            raise KeyError("Missing 'grid' mapping in mapping_file")
        self._mapping = TabularMapping(cast(Tables, mapping["grid"]))
        self._units = UnitMapping(cast(Units, mapping["units"])) if "units" in mapping else None
        self._substitutions = (
            ValueMapping(cast(Values, mapping["substitutions"])) if "substitutions" in mapping else None
        )

    def _parse_data(self, data: TabularData, data_type: str, extra_info: Optional[ExtraInfoLookup] = None) -> Dataset:
        data.set_units(self._units)
        data.set_substitutions(self._substitutions)
        input_data, extra = convert_excel_to_pgm(
            workbook=data, mapping=self._mapping.get_mapping() if self._mapping else {}
        )
        if extra_info is not None:
            extra_info.update(extra)
        self._log.debug("Converted tabular data to power grid model data")

        return input_data

    def _serialize_data(self, data: Dataset, extra_info: Optional[ExtraInfoLookup] = None) -> TabularData:
        if extra_info is not None:
            raise NotImplementedError("Extra info can not (yet) be stored for tabular data")
        if isinstance(data, list):
            raise NotImplementedError("Batch data can not(yet) be stored for tabular data")
        return data
