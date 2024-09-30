# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
These functions can be used in the mapping files to apply functions to vision data
"""

import math

import structlog
from power_grid_model import WindingType

from power_grid_model_io.functions import get_winding
from power_grid_model_io.utils.parsing import parse_pvs_efficiency_type, parse_trafo3_connection, parse_trafo_connection

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


def power_wind_speed(  # pylint: disable=too-many-arguments,too-many-positional-arguments
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


def _get_winding(trafo_connection_parser, winding_ref: str):
    def _get_winding_impl(conn_str: str, neutral_grounding: bool = True) -> WindingType:
        """
        Get the winding type, based on a textual encoding of the conn_str
        """
        return get_winding(trafo_connection_parser(conn_str)[winding_ref], neutral_grounding=neutral_grounding)

    return _get_winding_impl


def _get_clock(trafo_connection_parser, clock_ref: str):
    def _get_clock_impl(conn_str: str) -> int:
        """
        Extract the clock part of the conn_str
        """
        return int(trafo_connection_parser(conn_str)[clock_ref])

    return _get_clock_impl


get_winding_from = _get_winding(parse_trafo_connection, "winding_from")
get_winding_to = _get_winding(parse_trafo_connection, "winding_to")
get_winding_1 = _get_winding(parse_trafo3_connection, "winding_1")
get_winding_2 = _get_winding(parse_trafo3_connection, "winding_2")
get_winding_3 = _get_winding(parse_trafo3_connection, "winding_3")


get_clock = _get_clock(parse_trafo_connection, "clock")
get_clock_12 = _get_clock(parse_trafo3_connection, "clock_12")
get_clock_13 = _get_clock(parse_trafo3_connection, "clock_13")


def reactive_power_to_susceptance(q: float, u_nom: float) -> float:
    """
    Calculate susceptance, b1 from reactive power Q with nominal voltage
    """
    return q / u_nom / u_nom


def pvs_power_adjustment(p: float, efficiency_type: str) -> float:
    """
    Adjust power of PV for the default efficiency type of 97% or 95%. Defaults to 100 % for other custom types
    """
    try:
        pvs_efficiency_type = parse_pvs_efficiency_type(efficiency_type)
    except ValueError:
        return p

    _LOG.warning("PV approximation applied for efficiency type", efficiency_type=efficiency_type)
    if pvs_efficiency_type == "97":
        return p * 0.97
    if pvs_efficiency_type == "95":
        return p * 0.95

    return p
