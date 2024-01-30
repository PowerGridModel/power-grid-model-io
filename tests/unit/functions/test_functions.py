# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
from typing import List
from unittest.mock import MagicMock, patch

import numpy as np
from power_grid_model import WindingType
from pytest import approx, mark

from power_grid_model_io.functions import (
    both_zeros_to_nan,
    complex_inverse_imaginary_part,
    complex_inverse_real_part,
    degrees_to_clock,
    get_winding,
    has_value,
    is_greater_than,
    value_or_default,
    value_or_zero,
)


@mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        (0, True),
        (0.0, True),
        (float("nan"), False),
        (float("inf"), True),
        (float("-inf"), True),
        (1, True),
        (-(2**7), True),
        (-(2**31), True),
        ("", False),
        ("abc", True),
    ],
)
def test_has_value(value, expected):
    assert has_value(value) == approx(expected)


@patch("power_grid_model_io.functions.has_value")
def test_value_or_default__value(mock_has_value: MagicMock):
    mock_has_value.return_value = True
    assert value_or_default("value", "default") == "value"


@patch("power_grid_model_io.functions._functions.has_value")
def test_value_or_default__default(mock_has_value: MagicMock):
    mock_has_value.return_value = False
    assert value_or_default("value", "default") == "default"


@patch("power_grid_model_io.functions._functions.value_or_default")
def test_value_or_zero(mock_value_or_default: MagicMock):
    value_or_zero(1.23)
    mock_value_or_default.assert_called_once_with(value=1.23, default=0.0)


@mark.parametrize(
    ("real", "imag", "expected"),
    [
        (float("nan"), float("nan"), float("nan")),
        (1.0, 2.0, 0.2),
        (3.0, -1.0, 0.3),
        (2.0, 0.0, 0.5),
        (0.0, -4.0, 0.0),
    ],
)
def test_complex_inverse_real_part(real: float, imag: float, expected: float):
    actual = complex_inverse_real_part(real, imag)
    assert actual == approx(expected) or (np.isnan(actual) and np.isnan(expected))


@mark.parametrize(
    ("real", "imag", "expected"),
    [
        (float("nan"), float("nan"), float("nan")),
        (1.0, 2.0, -0.4),
        (3.0, -1.0, 0.1),
        (2.0, 0.0, 0.0),
        (0.0, -4.0, 0.25),
    ],
)
def test_complex_inverse_imaginary_part(real: float, imag: float, expected: float):
    actual = complex_inverse_imaginary_part(real, imag)
    assert actual == approx(expected) or (np.isnan(actual) and np.isnan(expected))


@mark.parametrize(
    ("winding", "neutral_grounding", "expected"),
    [
        ("Y", True, WindingType.wye),
        ("YN", True, WindingType.wye_n),
        ("D", True, WindingType.delta),
        ("Z", True, WindingType.zigzag),
        ("ZN", True, WindingType.zigzag_n),
        ("y", True, WindingType.wye),
        ("yn", True, WindingType.wye_n),
        ("d", True, WindingType.delta),
        ("z", True, WindingType.zigzag),
        ("zn", True, WindingType.zigzag_n),
        ("Y", False, WindingType.wye),
        ("YN", False, WindingType.wye),
        ("D", False, WindingType.delta),
        ("Z", False, WindingType.zigzag),
        ("ZN", False, WindingType.zigzag),
        ("y", False, WindingType.wye),
        ("yn", False, WindingType.wye),
        ("d", False, WindingType.delta),
        ("z", False, WindingType.zigzag),
        ("zn", False, WindingType.zigzag),
    ],
)
def test_get_winding(winding: str, neutral_grounding: bool, expected: WindingType):
    actual = get_winding(winding, neutral_grounding)
    assert actual == expected


@mark.parametrize(
    ("degrees", "expected"),
    [
        (360, 0),
        (120, 4),
        (180, 6),
        (540, 6),
    ],
)
def test_degrees_to_clock(degrees: float, expected: int):
    actual = degrees_to_clock(degrees)
    assert actual == approx(expected) or (np.isnan(actual) and np.isnan(expected))


@mark.parametrize(
    ("left_side", "right_side", "expected"),
    [
        (float("nan"), float("nan"), False),
        (0.0, 0.0, False),
        (1.0, 2.0, False),
        (2.0, 1.0, True),
    ],
)
def test_is_greater_than(left_side: float, right_side: List[float], expected: float):
    actual = is_greater_than(left_side, right_side)
    assert actual == expected


@mark.parametrize(
    ("value", "other_value", "expected"),
    [
        (float("nan"), float("nan"), float("nan")),
        (float("nan"), 0.0, float("nan")),
        (float("nan"), 5.0, float("nan")),
        (0.0, float("nan"), float("nan")),
        (0.0, 0.0, float("nan")),
        (0.0, 9.0, 0.0),
        (5.0, float("nan"), 5.0),
        (6.0, 0.0, 6.0),
        (7.0, 8.0, 7.0),
    ],
)
def test_both_zeros_to_nan(value: float, other_value: float, expected: float):
    actual = both_zeros_to_nan(value, other_value)
    assert actual == approx(expected) or (np.isnan(actual) and np.isnan(expected))
