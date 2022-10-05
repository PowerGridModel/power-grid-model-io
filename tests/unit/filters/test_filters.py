# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
from unittest.mock import MagicMock, patch

import numpy as np
from pytest import approx, mark

from power_grid_model_io.filters import (
    complex_inverse_imaginary_part,
    complex_inverse_real_part,
    has_value,
    multiply,
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


@patch("power_grid_model_io.filters.has_value")
def test_value_or_default__default(mock_has_value: MagicMock):
    mock_has_value.return_value = False
    assert value_or_default("value", "default") == "default"


@patch("power_grid_model_io.filters.value_or_default")
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
