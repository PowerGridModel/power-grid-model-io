# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from pytest import mark, raises

from power_grid_model_io.utils.modules import get_function


def test_get_function__builtins():
    assert get_function("min") == min


def test_get_function__native():
    assert get_function("pytest.mark") == mark


def test_get_function__custom():
    from power_grid_model_io.functions import complex_inverse_real_part

    assert get_function("power_grid_model_io.functions.complex_inverse_real_part") == complex_inverse_real_part


def test_get_function__module_doesnt_exist():
    with raises(AttributeError, match=r"Module 'a\.b' does not exist \(tried to resolve function 'a\.b\.c'\)!"):
        assert get_function("a.b.c")


def test_get_function__function_doesnt_exist():
    with raises(
        AttributeError, match="Function 'unknown_function' does not exist in module 'power_grid_model_io.functions'!"
    ):
        assert get_function("power_grid_model_io.functions.unknown_function")


def test_get_function__builtin_doesnt_exist():
    with raises(AttributeError, match="Function 'mean' does not exist in module 'builtins'!"):
        assert get_function("mean")
