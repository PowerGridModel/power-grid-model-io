# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

from power_grid_model import PowerGridModel

from power_grid_model_io.converters.pgm_converter import PgmConverter

ROOT = Path(__file__).parent.parent

if __name__ == "__main__":
    input_file = ROOT / "data" / "1os2msr" / "input.json"
    output_file = ROOT / "data" / "1os2msr" / "sym_output.json"

    converter = PgmConverter(source_file=input_file, destination_file=output_file)
    input_data, extra_info = converter.load_input_data()
    pgm = PowerGridModel(input_data=input_data)
    output_data = pgm.calculate_state_estimation()
    converter.save(data=output_data, extra_info=extra_info)

    print("Nodes output:", output_data["node"])
