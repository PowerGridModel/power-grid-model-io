# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import logging

import pandapower.networks
import pandas as pd
import structlog
from power_grid_model import PowerGridModel

from power_grid_model_io.converters.pandapower_converter import PandaPowerConverter
from power_grid_model_io.converters.pgm_json_converter import PgmJsonConverter
from power_grid_model_io.data_types.tabular_data import TabularData

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))

# Source data
net = pandapower.networks.example_simple()
pp_data = TabularData(**{component: net[component] for component in net if isinstance(net[component], pd.DataFrame)})

# Convert Vision file
pp_converter = PandaPowerConverter()
input_data, extra_info = pp_converter.load_input_data(pp_data)

# Store the source data in JSON format
converter = PgmJsonConverter(destination_file="pp_input.json")
converter.save(data=input_data, extra_info=extra_info)

# Perform power flow calculation
grid = PowerGridModel(input_data=input_data)
output_data = grid.calculate_power_flow()

# Store the result data in JSON format
converter = PgmJsonConverter(destination_file="pp_output.json")
converter.save(data=output_data, extra_info=extra_info)