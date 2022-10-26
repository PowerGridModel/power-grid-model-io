# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
These functions can be used in the mapping files to apply functions to pandapower data
"""
import re
from typing import Tuple

from power_grid_model import WindingType

from power_grid_model_io.filters import get_winding

CONNECTION_PATTERN_PP = re.compile(r"(Y|YN|D|Z|ZN)(y|yn|d|z|zn)\d*")
CONNECTION_PATTERN_PP_3WDG = re.compile(r"(Y|YN|D|Z|ZN)(y|yn|d|z|zn)(y|yn|d|z|zn)\d*")


def positive_sequence_conductance(power: float, voltage: float) -> float:
    """
    Calculate positive sequence conductance as used in shunts
    """
    return power / (voltage * voltage)


def get_transformer_clock(shift_degree: float) -> int:
    """
    Calculate the clock of a transformer
    """
    return int(shift_degree / 30)


def get_transformer_tap_size(
    high_side_voltage: float, low_side_voltage: float, tap_step_percent: float, tap_side: int
) -> float:
    """
    Calculate the tap_size of a transformer
    """
    if tap_side == 0:
        return (tap_step_percent / 100.0) * high_side_voltage
    if tap_side == 1:
        return (tap_step_percent / 100.0) * low_side_voltage
    raise ValueError(f"Only tap_side 0 and 1 are allowed, got {tap_side}.")


def get_3wdgtransformer_tap_size(
    high_side_voltage: float, med_side_voltage: float, low_side_voltage: float, tap_step_percent: float, tap_side: int
) -> float:
    """
    Calculate the tap_size of a three winding transformer
    """
    if tap_side == 0:
        return (tap_step_percent / 100.0) * high_side_voltage
    if tap_side == 1:
        return (tap_step_percent / 100.0) * med_side_voltage
    if tap_side == 2:
        return (tap_step_percent / 100.0) * low_side_voltage
    raise ValueError(f"Only tap_side 0, 1 and 2 are allowed, got {tap_side}.")


def _split_string(value: str) -> Tuple[str, str]:
    """
    Split the string of vector_group from PP into winding_from and winding_to of PGM
    """
    match = CONNECTION_PATTERN_PP.fullmatch(value)
    if not match:
        raise ValueError(f"Invalid transformer connection string: '{value}'")
    return match.group(1), match.group(2)


def get_transformer_winding_from(vector_group: str) -> WindingType:
    """
    Extract winding_from of a transformer
    """
    the_tuple = _split_string(vector_group)
    return get_winding(the_tuple[0])


def get_transformer_winding_to(vector_group: str) -> WindingType:
    """
    Extract winding_to of a transformer
    """
    the_tuple = _split_string(vector_group)
    return get_winding(the_tuple[1])


def _split_string_3wdg(value: str) -> Tuple[str, str, str]:
    """
    Split the string of vector_group from PP into winding_1, winding_2 and winding_3 of PGM
    """
    match = CONNECTION_PATTERN_PP_3WDG.fullmatch(value)
    if not match:
        raise ValueError(f"Invalid transformer connection string: '{value}'")
    return match.group(1), match.group(2), match.group(3)


def get_3wdgtransformer_winding_1(vector_group: str) -> WindingType:
    """
    Extract winding_1 of a three winding transformer
    """
    the_tuple = _split_string_3wdg(vector_group)
    return get_winding(the_tuple[0])


def get_3wdgtransformer_winding_2(vector_group: str) -> WindingType:
    """
    Extract winding_2 of a three winding transformer
    """
    the_tuple = _split_string_3wdg(vector_group)
    return get_winding(the_tuple[1])


def get_3wdgtransformer_winding_3(vector_group: str) -> WindingType:
    """
    Extract winding_3 of a three winding transformer
    """
    the_tuple = _split_string_3wdg(vector_group)
    return get_winding(the_tuple[2])


def get_3wdgtransformer_pk(percent: int, apparent_power1: float, apparent_power2: float) -> float:
    """
    Calculate pk of a three winding transformer
    """
    if apparent_power1 <= apparent_power2:
        return percent * apparent_power1
    if apparent_power1 > apparent_power2:
        return percent * apparent_power2
    return 0.0


def is_bus_switch(element_type: str) -> bool:  # is something's switch
    """
    In PP switches, elements of type 'b' are busses
    """
    return element_type == "b"
