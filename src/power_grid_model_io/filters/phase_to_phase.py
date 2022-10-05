# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
These functions can be used in the mapping files to apply functions to vision data
"""

import math
import re
from typing import Optional, Tuple, cast

from power_grid_model import WindingType

from power_grid_model_io.filters import has_value

CONNECTION_PATTERN = re.compile(r"(Y|YN|D|Z|ZN)(y|yn|d|z|zn)(\d|1[0-2])")

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


def relative_no_load_current(i: float, p: float, s_nom: float, u_nom: float) -> float:
    """
    TODO: description
    """
    return i / (s_nom / u_nom / math.sqrt(3)) if i > p / s_nom else p / s_nom


def reactive_power_calculation(pref: float, cosphi: float, scale: float) -> float:
    """
    Calculate the reactive power, based on Pref, cosine Phy and a scaling factor.
    """
    return scale * pref * math.sqrt((1 - math.pow(cosphi, 2) / cosphi))


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