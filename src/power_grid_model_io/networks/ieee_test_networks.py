# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
IEEE Test Networks loader
"""

from pathlib import Path

from power_grid_model.utils import json_deserialize_from_file

from power_grid_model_io.networks.enum import IEEE_four_bus_scenario

NETWORK_DATA_DIR = Path(__file__).parent / "data" / "ieee"


def ieee_four_bus(scenario: str | IEEE_four_bus_scenario = IEEE_four_bus_scenario.step_down_dyn1_balanced_load) -> dict:
    """

    Args:
        scenario (IEEE_four_bus_scenario | str): IEEE Four bus scenario, defaults to step-down Dyn1 balanced load.
    """

    if scenario in IEEE_four_bus_scenario:
        return json_deserialize_from_file(NETWORK_DATA_DIR / "ieee_four_bus" / f"input_{scenario!s}.json")

    raise ValueError("Invalid Scenario")
