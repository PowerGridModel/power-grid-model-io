# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import logging

import pandapower.networks
import pandas as pd
import structlog
from power_grid_model import PowerGridModel
from power_grid_model.validation import errors_to_string, validate_input_data

from power_grid_model_io.converters.pandapower_converter import PandaPowerConverter
from power_grid_model_io.converters.pgm_json_converter import PgmJsonConverter
from power_grid_model_io.data_types.tabular_data import TabularData

structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))

# Source data
pp_net = pandapower.networks.example_simple()
for col in pp_net["trafo3w"].columns:
    print(col)
print(pp_net)
pp_net_dict = {component: pp_net[component] for component in pp_net if isinstance(pp_net[component], pd.DataFrame)}
pp_data = TabularData(**pp_net_dict)

# Convert Vision file
pp_converter = PandaPowerConverter()
input_data, extra_info = pp_converter.load_input_data(pp_data)

print(errors_to_string(validate_input_data(input_data), details=True, id_lookup=extra_info))

# Store the source data in JSON format
converter = PgmJsonConverter(destination_file="data/pandapower/example_simple_input.json")
converter.save(data=input_data, extra_info=extra_info)

# Perform power flow calculation
grid = PowerGridModel(input_data=input_data)
output_data = grid.calculate_power_flow()

# Store the result data in JSON format
converter = PgmJsonConverter(destination_file="data/pandapower/example_simple_output.json")
converter.save(data=output_data, extra_info=extra_info)
