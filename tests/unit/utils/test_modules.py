# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from power_grid_model_io.functions import complex_inverse_real_part
from power_grid_model_io.utils.modules import get_function


def test_get_function__builtins():
    with pytest.raises(AttributeError, match=r"'min' is not an allowed mapping function"):
        assert get_function("min")


def test_get_function__native():
    with pytest.raises(AttributeError, match=r"'pytest\.mark' is not an allowed mapping function"):
        assert get_function("pytest.mark")


def test_get_function__custom():
    assert get_function("power_grid_model_io.functions.complex_inverse_real_part") == complex_inverse_real_part


def test_get_function__module_doesnt_exist():
    with pytest.raises(AttributeError, match=r"'a\.b\.c' is not an allowed mapping function"):
        assert get_function("a.b.c")


def test_get_function__function_doesnt_exist():
    with pytest.raises(
        AttributeError, match=r"'power_grid_model_io\.functions\.unknown_function' is not an allowed mapping function"
    ):
        assert get_function("power_grid_model_io.functions.unknown_function")


def test_get_function__builtin_doesnt_exist():
    with pytest.raises(AttributeError, match=r"'mean' is not an allowed mapping function"):
        assert get_function("mean")


def test_get_function__known():
    assert get_function("power_grid_model_io.functions.complex_inverse_real_part") is complex_inverse_real_part


def test_get_function__unknown():
    with pytest.raises(AttributeError, match=r"'no\.such\.fn' is not an allowed mapping function"):
        get_function("no.such.fn")
