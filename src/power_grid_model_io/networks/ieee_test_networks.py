# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
IEEE Test Networks loader
"""

from pathlib import Path

from power_grid_model.data_types import Dataset
from power_grid_model.utils import json_deserialize_from_file

from power_grid_model_io.networks.enum import IeeeFourNodeVariant

NETWORK_DATA_DIR = Path(__file__).parent / "_data" / "ieee"


def ieee_four_node(variant: str | IeeeFourNodeVariant = IeeeFourNodeVariant.step_down_dyn_balanced_loading) -> Dataset:
    """
    Load pre-built IEEE 4-node standard 2-winding test feeder.

    Args:
        variant: (str | IeeeFourBusVariant): IEEE Four node network variant, defaults to step_down_dyn_balanced_loading.
    """

    if variant in IeeeFourNodeVariant:
        return json_deserialize_from_file(NETWORK_DATA_DIR / "ieee_four_node" / str(variant) / "input.json")

    raise ValueError(f"Invalid variant: {variant!r}. Not a recognized IeeeFourBusVariant.")
