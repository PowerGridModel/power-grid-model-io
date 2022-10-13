# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
These functions can be used in the mapping files to apply functions to tabular data
"""

import math
import re
from typing import Any, Optional, Tuple, TypeVar, cast

import numpy as np
from power_grid_model import WindingType

CONNECTION_PATTERN = re.compile(r"(Y|YN|D|Z|ZN)(y|yn|d|z|zn)(\d|1[0-2])")
CONNECTION_PATTERN_PP = re.compile(r"(Y|YN|D|Z|ZN)(y|yn|d|z|zn)")
CONNECTION_PATTERN_PP_3WDG = re.compile(r"(Y|YN|D|Z|ZN)(y|yn|d|z|zn)(y|yn|d|z|zn)")

"""
TODO: Implement Z winding
"""
WINDING_TYPES = {
    "Y": WindingType.wye,
    "YN": WindingType.wye_n,
    "D": WindingType.delta,
    "Z": WindingType.wye,
    "ZN": WindingType.wye_n,
}

T = TypeVar("T")


def relative_no_load_current(i: float, p: float, s_nom: float, u_nom: float) -> float:
    """
    TODO: description
    """
    return i / (s_nom / u_nom / math.sqrt(3)) if i > p / s_nom else p / s_nom


def multiply(*args: float):
    """
    Multiply all arguments.
    """
    return math.prod(args)


def reactive_power_calculation(pref: float, cosphi: float, scale: float) -> float:
    """
    Calculate the reactive power, based on Pref, cosine Phy and a scaling factor.
    """
    return scale * pref * math.sqrt((1 - math.pow(cosphi, 2) / cosphi))


def has_value(value: Any) -> bool:
    """
    Return True if the value is not None or NaN.
    """
    return value is not None and not np.isnan(value)


def value_or_default(value: Optional[T], default: T) -> T:
    """
    Return the value, or a default value if no value was supplied.
    """
    return cast(T, value) if has_value(value) else default


def value_or_zero(value: Optional[float]) -> float:
    """
    Return the value, or a zero value if no value was supplied.
    """
    return value_or_default(value=value, default=0.0)


def power_wind_speed(pref: Optional[float], pnom: float, v: float) -> float:
    """
    Return Pref is available, otherwise estimate Pref based on Pnom and v.
    TODO: Add a reference for the calculations
    """
    if has_value(pref):
        return cast(float, pref)
    if v < 3:
        return 0.0
    if v < 14:
        return pnom * (math.pow(v, 3) / math.pow(14, 3))
    if v < 25:
        return pnom
    if v < 30:
        return pnom * (1 - (v - 25) / (30 - 25))
    return 0.0


def complex_inverse_real_part(real: float, imag: float) -> float:
    """
    Return the real part of the inverse of a complex number
    """
    return (1.0 / (real + 1j * imag)).real


def complex_inverse_imaginary_part(real: float, imag: float) -> float:
    """
    Return the imaginary part of the inverse of a complex number
    """
    return (1.0 / (real + 1j * imag)).imag


def get_winding_from(conn_str: str, neutral_grounding: bool = True) -> WindingType:
    """
    Get the winding type, based on a textual encoding of the conn_str
    TODO: use z winding when zigzag is implemented
    """
    wfr, wto, clock = _split_connection_string(conn_str)
    winding = WINDING_TYPES[wfr]
    if winding == WindingType.wye_n and not neutral_grounding:
        winding = WindingType.wye
    if wfr[0] == "Z" and wto != "d" and clock % 2:
        winding = WindingType.delta
    return winding


def get_winding_to(conn_str: str, neutral_grounding: bool = True) -> WindingType:
    """
    Get the winding type, based on a textual encoding of the conn_str
    TODO: use z winding when zigzag is implemented
    """
    wfr, wto, clock = _split_connection_string(conn_str)
    winding = WINDING_TYPES[wto.upper()]
    if winding == WindingType.wye_n and not neutral_grounding:
        winding = WindingType.wye
    if wfr != "D" and wto[0] == "z" and clock % 2:
        winding = WindingType.delta
    return winding


def get_clock(conn_str: str) -> int:
    """
    Extract the clock part of the conn_str
    """
    _, _, clock = _split_connection_string(conn_str)
    return clock


def _split_connection_string(conn_str: str) -> Tuple[str, str, int]:
    """
    Helper function to split the conn_str into three parts:
     * winding_from
     * winding_to
     * clock
    """
    match = CONNECTION_PATTERN.fullmatch(conn_str)
    if not match:
        raise ValueError(f"Invalid transformer connection string: '{conn_str}'")
    return match.group(1), match.group(2), int(match.group(3))


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
    high_side_voltage: float, low_side_voltage: float, tap_step_percent: float, tap_side: str
) -> float:  # result should be double?
    """
    Calculate the tap_size of a transformer
    """
    if tap_side == "hv":
        return tap_step_percent * high_side_voltage
    elif tap_side == "lv":
        return tap_step_percent * low_side_voltage


def get_3wdgtransformer_tap_size(
    high_side_voltage: float, med_side_voltage: float, low_side_voltage: float, tap_step_percent: float, tap_side: str
) -> float:
    """
    Calculate the tap_size of a three winding transformer
    """
    if tap_side == "hv":
        return tap_step_percent * high_side_voltage
    elif tap_side == "mv":
        return tap_step_percent * med_side_voltage
    elif tap_side == "lv":
        return tap_step_percent * low_side_voltage


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
    Extract winding_from of a transfomer
    """
    the_tuple = _split_string(vector_group)
    winding = WINDING_TYPES[the_tuple[0]]
    if winding == WindingType.wye:
        winding = WindingType.wye
    elif winding == WindingType.wye_n:
        winding = WindingType.wye_n
    elif winding == WindingType.delta:
        winding = WindingType.delta
    elif winding == WindingType.zigzag:
        winding = WindingType.zigzag
    elif winding == WindingType.zigzag_n:
        winding = WindingType.zigzag_n
    return winding


def get_transformer_winding_to(vector_group: str) -> WindingType:
    """
    Extract winding_to of a transformer
    """
    the_tuple = _split_string(vector_group)
    winding = WINDING_TYPES[the_tuple[1].upper()]
    if winding == WindingType.wye:
        winding = WindingType.wye
    elif winding == WindingType.wye_n:
        winding = WindingType.wye_n
    elif winding == WindingType.delta:
        winding = WindingType.delta
    elif winding == WindingType.zigzag:
        winding = WindingType.zigzag
    elif winding == WindingType.zigzag_n:
        winding = WindingType.zigzag_n
    return winding


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
    winding = WINDING_TYPES[the_tuple[0]]
    if winding == WindingType.wye:
        winding = WindingType.wye
    elif winding == WindingType.wye_n:
        winding = WindingType.wye_n
    elif winding == WindingType.delta:
        winding = WindingType.delta
    elif winding == WindingType.zigzag:
        winding = WindingType.zigzag
    elif winding == WindingType.zigzag_n:
        winding = WindingType.zigzag_n
    return winding


def get_3wdgtransformer_winding_2(vector_group: str) -> WindingType:
    """
    Extract winding_2 of a three winding transformer
    """
    the_tuple = _split_string_3wdg(vector_group)
    winding = WINDING_TYPES[the_tuple[1].upper()]
    if winding == WindingType.wye:
        winding = WindingType.wye
    elif winding == WindingType.wye_n:
        winding = WindingType.wye_n
    elif winding == WindingType.delta:
        winding = WindingType.delta
    elif winding == WindingType.zigzag:
        winding = WindingType.zigzag
    elif winding == WindingType.zigzag_n:
        winding = WindingType.zigzag_n
    return winding


def get_3wdgtransformer_winding_3(vector_group: str) -> WindingType:
    """
    Extract winding_3 of a three winding transformer
    """
    the_tuple = _split_string_3wdg(vector_group)
    winding = WINDING_TYPES[the_tuple[2].upper()]
    if winding == WindingType.wye:
        winding = WindingType.wye
    elif winding == WindingType.wye_n:
        winding = WindingType.wye_n
    elif winding == WindingType.delta:
        winding = WindingType.delta
    elif winding == WindingType.zigzag:
        winding = WindingType.zigzag
    elif winding == WindingType.zigzag_n:
        winding = WindingType.zigzag_n
    return winding
