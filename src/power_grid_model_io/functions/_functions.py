# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""
These functions can be used in the mapping files to apply functions to tabular data
"""

from typing import Any, Optional, TypeVar, cast

import numpy as np
import structlog
from power_grid_model import MeasuredTerminalType, WindingType

T = TypeVar("T")

_LOG = structlog.get_logger(__file__)

WINDING_TYPES = {
    "Y": WindingType.wye,
    "YN": WindingType.wye_n,
    "D": WindingType.delta,
    "Z": WindingType.zigzag,
    "ZN": WindingType.zigzag_n,
}

MEASURED_TERMINAL_TYPE_MAP = {
    "cable_from": MeasuredTerminalType.branch_from,
    "cable_to": MeasuredTerminalType.branch_to,
    "line_from": MeasuredTerminalType.branch_from,
    "line_to": MeasuredTerminalType.branch_to,
    "reactance_coil_from": MeasuredTerminalType.branch_from,
    "reactance_coil_to": MeasuredTerminalType.branch_to,
    "special_transformer_from": MeasuredTerminalType.branch_from,
    "special_transformer_to": MeasuredTerminalType.branch_to,
    "transformer_from": MeasuredTerminalType.branch_from,
    "transformer_to": MeasuredTerminalType.branch_to,
    "transformer_load": MeasuredTerminalType.branch_to,
    "earthing_transformer": MeasuredTerminalType.branch_from,
    "transformer3_1": MeasuredTerminalType.branch3_1,
    "transformer3_2": MeasuredTerminalType.branch3_2,
    "transformer3_3": MeasuredTerminalType.branch3_3,
    "source": MeasuredTerminalType.source,
    "shunt_capacitor": MeasuredTerminalType.shunt,
    "shunt_reactor": MeasuredTerminalType.shunt,
    "pv": MeasuredTerminalType.generator,
    "wind_turbine": MeasuredTerminalType.generator,
    "load": MeasuredTerminalType.load,
}


def has_value(value: Any) -> bool:
    """
    Return True if the value is not None, NaN or empty string.
    """
    if value is None:
        return False
    if isinstance(value, float):
        return not np.isnan(value)
    return value != ""


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


def get_winding(winding: str, neutral_grounding: bool = True) -> WindingType:
    """
    Return the winding type as an enum value, based on the string representation
    """
    winding_type = WINDING_TYPES[winding.upper()]
    if not neutral_grounding:
        if winding_type == WindingType.wye_n:
            return WindingType.wye
        if winding_type == WindingType.zigzag_n:
            return WindingType.zigzag
    return winding_type


def degrees_to_clock(degrees: float) -> int:
    """
    Return the clock
    """
    return int(round(degrees / 30.0)) % 12


def is_greater_than(left_side, right_side) -> bool:
    """
    Return true if the first argument is greater than the second
    """
    return left_side > right_side


def both_zeros_to_nan(value: float, other_value: float) -> float:
    """
    If both values are zero then return nan otherwise return same value.
    Truth table (x = value, y = other_value)
             0     value     nan
    0       nan    value     nan
    value   0      value     nan
    nan     nan    value     nan
    """
    if value == 0 and (other_value == 0 or not has_value(other_value)):
        _LOG.warning("0 replaced to nan")
        return float("nan")
    return value


def find_terminal_type(**kwargs) -> MeasuredTerminalType:
    """
    Return the measured terminal type, based on the string representation
    """
    for key, id in kwargs.items():
        if id is not None:
            return MEASURED_TERMINAL_TYPE_MAP[key]
    _LOG.warning("No measured terminal type is found!")
    return float("nan")

def filter_if_object(object_name: str, excl_object: str) -> bool:
    """
    Return false if the measured object should be excluded.
    """
    return object_name != excl_object
