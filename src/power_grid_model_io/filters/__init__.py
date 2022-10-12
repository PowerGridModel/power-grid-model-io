# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""
These functions can be used in the mapping files to apply functions to tabular data
"""

import math
from typing import Any, Optional, TypeVar, cast

import numpy as np

T = TypeVar("T")


def multiply(*args: float):
    """
    Multiply all arguments.
    """
    return math.prod(args)


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
