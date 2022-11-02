# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
from typing import List, Tuple, Union
from unittest.mock import MagicMock, patch

import numpy as np
from power_grid_model import WindingType
from pytest import approx, mark, param

from power_grid_model_io.filters import (
    all_true,
    any_true,
    complex_inverse_imaginary_part,
    complex_inverse_real_part,
    degrees_to_clock,
    get_winding,
    has_value,
    is_greater_than,
    multiply,
    subtract,
    value_or_default,
    value_or_zero,
)


@mark.parametrize(
    ("args", "expected"),
    [
        ([], 1),
        ([2], 2),
        ([2, 3], 6),
        ([2, 3, 5], 30),
        ([2.0, 3.0, 5.0, 7.0], 210.0),
    ],
)
def test_multiply_0(args, expected):
    assert multiply(*args) == approx(expected)


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


@patch("power_grid_model_io.filters.has_value")
def test_value_or_default__value(mock_has_value: MagicMock):
    mock_has_value.return_value = True
    assert value_or_default("value", "default") == "value"


@patch("power_grid_model_io.filters._filters.has_value")
def test_value_or_default__default(mock_has_value: MagicMock):
    mock_has_value.return_value = False
    assert value_or_default("value", "default") == "default"


@patch("power_grid_model_io.filters._filters.value_or_default")
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
    ("value", "arguments", "expected"),
    [
        (360.0, [0.0, 10.0, 300.0], 50.0),
        (120.0, [4.0, 15.5, 63.5], 37.0),
        (180.0, [10.5, 20.5, 30.5, 40.5], 78.0),
        (540.0, [50.5, 10.5, 101.0, 64.5, 3.5], 310.0),
    ],
)
def test_subtract(value: float, arguments: List[float], expected: float):
    actual = subtract(value, *arguments)
    assert actual == approx(expected) or (np.isnan(actual) and np.isnan(expected))


@mark.parametrize(
    ("args", "expected"),
    [
        param(tuple(), True, id="no_args"),
        param((True,), True, id="single_true"),
        param((False,), False, id="single_false"),
        param((True, True), True, id="only_true"),
        param((True, False), False, id="mixed_true_false"),
        param((False, False), False, id="only_false"),
        param((True, 1, 1.0), True, id="mixed_types"),
        param((True, 1, 0.0), False, id="mixed_types_zero_float"),
        param((True, 0, 1.0), False, id="mixed_types_zero_int"),
        param((False, 1, 1.0), False, id="mixed_types_false_bool"),
        param((True, 1, np.nan), False, id="mixed_types_nan_float"),
    ],
)
def test_all_true(args: Tuple[Union[bool, int, float], ...], expected: bool):
    actual = all_true(*args)
    assert actual == expected


@mark.parametrize(
    ("args", "expected"),
    [
        param(tuple(), False, id="no_args"),
        param((True,), True, id="single_true"),
        param((False,), False, id="single_false"),
        param((True, True), True, id="only_true"),
        param((True, False), True, id="mixed_true_false"),
        param((False, False), False, id="only_false"),
        param((False, 0, 0.0), False, id="mixed_types"),
        param((False, 0, 1.0), True, id="mixed_types_non_zero_float"),
        param((False, 1, 0.0), True, id="mixed_types_non_zero_int"),
        param((True, 0, 0.0), True, id="mixed_types_true_bool"),
        param((False, 0, np.nan), False, id="mixed_types_nan_float"),
    ],
)
def test_any_true(args: Tuple[Union[bool, int, float], ...], expected: bool):
    actual = any_true(*args)
    assert actual == expected


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
