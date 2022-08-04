# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path

from power_grid_model.data_types import Dataset, ExtraInfo
from power_grid_model.utils import export_json_data

from power_grid_model_io.converters.base_converter import BaseConverter


class PgmConverter(BaseConverter):
    def save_data(self, data: Dataset, extra_info: ExtraInfo, dst: Path):
        # Check JSON file name
        if dst.suffix.lower() != ".json":
            raise ValueError(f"Output file should be a .json file, {dst.suffix} provided.")

        # Store JSON data
        self._log.debug("Writing PGM JSON file", dst_file=dst)
        export_json_data(json_file=dst, data=data, extra_info=extra_info, indent=2, compact=True)
