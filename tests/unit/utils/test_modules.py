# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock, patch

from pytest import mark, raises

from power_grid_model_io.utils.modules import (
    DEPENDENCIES,
    assert_optional_module_installed,
    get_function,
    import_optional_module,
    module_installed,
    running_from_conda,
)


@mark.parametrize("conda_exists", [True, False])
@patch("power_grid_model_io.utils.modules.Path.exists")
def test_running_from_conda(exists_mock: MagicMock, conda_exists: bool):
    exists_mock.return_value = conda_exists
    assert running_from_conda() == conda_exists


def test_module_installed():
    assert module_installed("power_grid_model_io")
    assert not module_installed("non_existing_module")


@patch("importlib.import_module")
@patch("power_grid_model_io.utils.modules.assert_optional_module_installed")
def test_import_optional_module(assert_mock: MagicMock, import_mock: MagicMock):
    module = import_optional_module(module="module", extra="extra")
    assert_mock.assert_called_once_with(module="module", extra="extra")
    import_mock.assert_called_once_with("module")
    assert module == import_mock.return_value


@patch("power_grid_model_io.utils.modules.module_installed")
def test_assert_optional_module_installed__ok(module_installed_mock: MagicMock):
    DEPENDENCIES["dummy_extra"] = {"dummy_module": "dummy_package"}
    module_installed_mock.return_value = True
    assert_optional_module_installed(module="dummy_module", extra="dummy_extra")


def test_assert_optional_module_installed__unknown_extra():
    DEPENDENCIES["dummy_extra"] = {"dummy_module": "dummy_package"}
    with raises(KeyError, match="unknown_extra"):
        assert_optional_module_installed(module="dummy_module", extra="unknown_extra")


def test_assert_optional_module_installed__unknown_module():
    DEPENDENCIES["dummy_extra"] = {"dummy_module": "dummy_package"}
    with raises(KeyError, match="unknown_module.*dummy_extra"):
        assert_optional_module_installed(module="unknown_module", extra="dummy_extra")


@patch("power_grid_model_io.utils.modules.running_from_conda")
def test_assert_optional_module_installed__conda(conda_mock: MagicMock):
    conda_mock.return_value = True
    DEPENDENCIES["dummy_extra"] = {"dummy_module": "dummy_package"}
    with raises(
        ModuleNotFoundError, match=r"Missing.*dummy_module.*`conda install power-grid-model-io\[dummy_extra\]`"
    ):
        assert_optional_module_installed(module="dummy_module", extra="dummy_extra")


@patch("power_grid_model_io.utils.modules.running_from_conda")
@patch("power_grid_model_io.utils.modules.module_installed")
def test_assert_optional_module_installed__pip(module_mock: MagicMock, conda_mock: MagicMock):
    module_mock.side_effect = [False, True]  # dummy_module: False, pip: True
    conda_mock.return_value = False
    DEPENDENCIES["dummy_extra"] = {"dummy_module": "dummy_package"}
    with raises(ModuleNotFoundError, match=r"Missing.*dummy_module.*`pip install power-grid-model-io\[dummy_extra\]`"):
        assert_optional_module_installed(module="dummy_module", extra="dummy_extra")


@patch("power_grid_model_io.utils.modules.running_from_conda")
@patch("power_grid_model_io.utils.modules.module_installed")
def test_assert_optional_module_installed__unkown_package_manager(module_mock: MagicMock, conda_mock: MagicMock):
    module_mock.side_effect = [False, False]  # dummy_module: False, pip: False
    conda_mock.return_value = False
    DEPENDENCIES["dummy_extra"] = {"dummy_module": "dummy_package"}
    with raises(ModuleNotFoundError, match=r"Missing.*dummy_module.*`power-grid-model-io\[dummy_extra\]`"):
        assert_optional_module_installed(module="dummy_module", extra="dummy_extra")


def test_get_function__builtins():
    assert get_function("min") == min


def test_get_function__native():
    assert get_function("pytest.mark") == mark


def test_get_function__custom():
    from power_grid_model_io.filters import multiply

    assert get_function("power_grid_model_io.filters.multiply") == multiply


def test_get_function__module_doesnt_exist():
    with raises(AttributeError, match=r"Module 'a\.b' does not exist \(tried to resolve function 'a\.b\.c'\)!"):
        assert get_function("a.b.c")


def test_get_function__function_doesnt_exist():
    with raises(
        AttributeError, match="Function 'unknown_function' does not exist in module 'power_grid_model_io.filters'!"
    ):
        assert get_function("power_grid_model_io.filters.unknown_function")


def test_get_function__builtin_doesnt_exist():
    with raises(AttributeError, match="Function 'mean' does not exist in module 'builtins'!"):
        assert get_function("mean")
