# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
IEEE Test Networks loader
"""

from pathlib import Path

from power_grid_model import DatasetType
from power_grid_model.utils import json_deserialize_from_file

NETWORK_DATA_DIR = Path(__file__).parent / "data" / "ieee"


def ieee_four_bus(
    data_type: DatasetType = DatasetType.input,
    symmetric_load: bool = True,
    trafo_type: str = "Dyn1",
) -> dict:
    """_summary_

    Args:
        type (DatasetType, optional): _description_. Defaults to DatasetType.input.
    """

    if (data_type == DatasetType.input) & (symmetric_load) & (trafo_type == "Dyn1"):
        return json_deserialize_from_file(NETWORK_DATA_DIR / "ieee_four_bus" / "input_step_down_dyn1_balanced.json")

    raise ValueError("Invalid Input")
