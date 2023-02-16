# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
These functions can be used in the mapping files to apply functions to vision data
"""

import math
from typing import Tuple

import structlog
from power_grid_model import WindingType

from power_grid_model_io.functions import get_winding
from power_grid_model_io.utils.regex import PVS_EFFICIENCY_TYPE_RE, TRAFO3_CONNECTION_RE, TRAFO_CONNECTION_RE

_LOG = structlog.get_logger(__file__)


def relative_no_load_current(i_0: float, p_0: float, s_nom: float, u_nom: float) -> float:
    """
    Calculate the relative no load current.
    """
    i_rel = max(i_0 / (s_nom / (u_nom * math.sqrt(3))), p_0 / s_nom)
    if i_rel > 1.0:
        raise ValueError(f"Relative current can't be more than 100% (got {i_rel * 100.0:.2f}%)")
    return i_rel


def reactive_power(p: float, cos_phi: float) -> float:
    """
    Calculate the reactive power, based on p, cosine phi.
    """
    return p * math.sqrt(1 - cos_phi**2) / cos_phi


def power_wind_speed(  # pylint: disable=too-many-arguments
    p_nom: float,
    wind_speed: float,
    cut_in_wind_speed: float = 3.0,
    nominal_wind_speed: float = 14.0,
    cutting_out_wind_speed: float = 25.0,
    cut_out_wind_speed: float = 30.0,
    axis_height: float = 30.0,
) -> float:
    """
    Estimate p_ref based on p_nom and wind_speed.

    See section "Wind turbine" in https://phasetophase.nl/pdf/VisionEN.pdf
    """

    # Calculate wind speed at the axis height
    wind_speed *= (axis_height / 10) ** 0.143

    # At a wind speed below cut-in, the power is zero.
    if wind_speed < cut_in_wind_speed:
        return 0.0

    # At a wind speed between cut-in and nominal, the power is a third power function of the wind speed.
    if wind_speed < nominal_wind_speed:
        factor = wind_speed - cut_in_wind_speed
        max_factor = nominal_wind_speed - cut_in_wind_speed
        return ((factor / max_factor) ** 3) * p_nom

    # At a wind speed between nominal and cutting-out, the power is the nominal power.
    if wind_speed < cutting_out_wind_speed:
        return p_nom

    # At a wind speed between cutting-out and cut-out, the power decreases from nominal to zero.
    if wind_speed < cut_out_wind_speed:
        factor = wind_speed - cutting_out_wind_speed
        max_factor = cut_out_wind_speed - cutting_out_wind_speed
        return (1.0 - factor / max_factor) * p_nom

    # Above cut-out speed, the power is zero.
    return 0.0


def get_winding_from(conn_str: str, neutral_grounding: bool = True) -> WindingType:
    """
    Get the winding type, based on a textual encoding of the conn_str
    """
    winding_from, _, _ = _split_connection_string(conn_str)
    return get_winding(winding=winding_from, neutral_grounding=neutral_grounding)


def get_winding_to(conn_str: str, neutral_grounding: bool = True) -> WindingType:
    """
    Get the winding type, based on a textual encoding of the conn_str
    """
    _, winding_to, _ = _split_connection_string(conn_str)
    return get_winding(winding=winding_to, neutral_grounding=neutral_grounding)


def get_winding_1(conn_str: str, neutral_grounding: bool = True) -> WindingType:
    """
    Get the winding type, based on a textual encoding of the conn_str
    """
    winding_1, _, _, _, _ = _split_connection_string_3w(conn_str)
    return get_winding(winding=winding_1, neutral_grounding=neutral_grounding)


def get_winding_2(conn_str: str, neutral_grounding: bool = True) -> WindingType:
    """
    Get the winding type, based on a textual encoding of the conn_str
    """
    _, winding_2, _, _, _ = _split_connection_string_3w(conn_str)
    return get_winding(winding=winding_2, neutral_grounding=neutral_grounding)


def get_winding_3(conn_str: str, neutral_grounding: bool = True) -> WindingType:
    """
    Get the winding type, based on a textual encoding of the conn_str
    """
    _, _, _, winding_3, _ = _split_connection_string_3w(conn_str)
    return get_winding(winding=winding_3, neutral_grounding=neutral_grounding)


def get_clock(conn_str: str) -> int:
    """
    Extract the clock part of the conn_str
    """
    _, _, clock = _split_connection_string(conn_str)
    return clock


def get_clock_12(conn_str: str) -> int:
    """
    Extract the clock part of the conn_str
    """
    _, _, clock_12, _, _ = _split_connection_string_3w(conn_str)
    return clock_12


def get_clock_13(conn_str: str) -> int:
    """
    Extract the clock part of the conn_str
    """
    _, _, _, _, clock_13 = _split_connection_string_3w(conn_str)
    return clock_13


def reactive_power_to_susceptance(q: float, u_nom: float) -> float:
    """
    Calculate susceptance, b1 from reactive power Q with nominal voltage
    """
    return q / u_nom / u_nom


def _split_connection_string(conn_str: str) -> Tuple[str, str, int]:
    """
    Helper function to split the conn_str into three parts:
     * winding_from
     * winding_to
     * clock
    """
    match = TRAFO_CONNECTION_RE.fullmatch(conn_str)
    if not match:
        raise ValueError(f"Invalid transformer connection string: '{conn_str}'")
    return match.group(1), match.group(2), int(match.group(3))


def _split_connection_string_3w(conn_str: str) -> Tuple[str, str, int, str, int]:
    """
    Helper function to split the conn_str into three parts:
     * winding_1
     * winding_2
     * clock 12
     * winding_3
     * clock 13
    """
    match = TRAFO3_CONNECTION_RE.fullmatch(conn_str)
    if not match:
        raise ValueError(f"Invalid three winding transformer connection string: '{conn_str}'")
    return match.group(1), match.group(2), int(match.group(3)), match.group(4), int(match.group(5))


def pvs_power_adjustment(p: float, efficiency_type: str) -> float:
    """
    Adjust power of PV for the default efficiency type of 97% or 95%. Defaults to 100 % for other custom types
    """
    match = PVS_EFFICIENCY_TYPE_RE.search(efficiency_type)
    if match is not None:
        _LOG.warning("PV approximation applied for efficiency type", efficiency_type=efficiency_type)
        if match.group(1) == "97":
            return p * 0.97
        if match.group(1) == "95":
            return p * 0.95
    return p
