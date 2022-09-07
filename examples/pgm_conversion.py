# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

from power_grid_model import PowerGridModel

from power_grid_model_io.converters.pgm_converter import PgmConverter
from power_grid_model_io.data_stores.json_file_store import JsonFileStore

ROOT = Path(__file__).parent.parent

if __name__ == "__main__":

    input_file = JsonFileStore(ROOT / "data" / "1os2msr" / "input.json")
    output_file = JsonFileStore(ROOT / "output.json")

    converter = PgmConverter()
    input_data, extra_info = converter.load_input_data(input_file)
    pgm = PowerGridModel(input_data=input_data)
    output_data = pgm.calculate_state_estimation()
    converter.save_data(output_file, output_data)

    print("Nodes output:", output_data["node"])
