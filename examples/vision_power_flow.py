# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import logging

import structlog
from power_grid_model import CalculationType, PowerGridModel
from power_grid_model.validation import assert_valid_input_data

from power_grid_model_io.converters.pgm_json_converter import PgmJsonConverter
from power_grid_model_io.converters.vision_excel_converter import VisionExcelConverter

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))

# Source and destination file
src = "data/vision/example.xlsx"
dst = "data/vision/example_output.json"

# Convert Vision file
vision_converter = VisionExcelConverter(source_file=src)
input_data, extra_info = vision_converter.load_input_data()

# Validate data
assert_valid_input_data(input_data, calculation_type=CalculationType.power_flow, symmetric=True)

# Perform power flow calculation
grid = PowerGridModel(input_data=input_data)
output_data = grid.calculate_power_flow()

# Store the results in JSON format
converter = PgmJsonConverter(destination_file=dst)
converter.save(data=output_data, extra_info=extra_info)
