# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
IEEE Test Networks loader
"""

from pathlib import Path

from power_grid_model.data_types import Dataset
from power_grid_model.utils import json_deserialize_from_file

from power_grid_model_io.networks.enum import IeeeFourBusScenario

NETWORK_DATA_DIR = Path(__file__).parent / "_data" / "ieee"


def ieee_four_bus(scenario: str | IeeeFourBusScenario = IeeeFourBusScenario.step_down_dyn_balanced_load) -> Dataset:
    """

    Args:
        scenario (IEEE_four_bus_scenario | str): IEEE Four bus scenario, defaults to step-down Dyn1 balanced load.
    """

    if scenario in IeeeFourBusScenario:
        return json_deserialize_from_file(NETWORK_DATA_DIR / "ieee_four_bus" / str(scenario) / "input.json")

    raise ValueError(f"Invalid scenario: {scenario!r}. Not a recognized IeeeFourBusScenario.")
