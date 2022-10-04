# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

from power_grid_model import PowerGridModel

from power_grid_model_io.converters.pgm_json_converter import PgmJsonConverter
from power_grid_model_io.converters.vision_excel_converter import VisionExcelConverter

# Source and destination file
src = Path("data/vision/example.xlsx")
dst = Path("data/vision/example_output.json")

# Convert Vision file
vision_converter = VisionExcelConverter(source_file=src)
input_data, extra_info = vision_converter.load_input_data()

# Perform power flow calculation
grid = PowerGridModel(input_data=input_data)
output_data = grid.calculate_power_flow()

# Store the results in JSON format
converter = PgmJsonConverter(destination_file=dst)
converter.save(data=output_data, extra_info=extra_info)
