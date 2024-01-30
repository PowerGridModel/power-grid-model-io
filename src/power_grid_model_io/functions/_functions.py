# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

"""
These functions can be used in the mapping files to apply functions to tabular data
"""

from typing import Any, Optional, TypeVar, cast

import numpy as np
import structlog
from power_grid_model import WindingType

T = TypeVar("T")

_LOG = structlog.get_logger(__file__)

WINDING_TYPES = {
    "Y": WindingType.wye,
    "YN": WindingType.wye_n,
    "D": WindingType.delta,
    "Z": WindingType.zigzag,
    "ZN": WindingType.zigzag_n,
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
