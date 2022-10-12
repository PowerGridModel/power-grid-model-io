# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Module utilities, expecially useful for loading optional dependencies
"""
import importlib.util
import sys
from importlib import import_module
from pathlib import Path
from types import ModuleType
from typing import Callable

MAIN_PACKAGE = "power-grid-model-io"

# extra_key -> module -> pip_package
DEPENDENCIES = {
    "cli": {"typer": "typer[all]"},
}


def running_from_conda() -> bool:
    """
    Check if the conda is used
    """
    # If conda is used, we expect a directory called conda-meta in the root dir of the environment
    env_dir = Path(sys.prefix)
    return (env_dir / "conda-meta").exists()


def module_installed(module: str) -> bool:
    """
    Check if the module is installed
    """
    return importlib.util.find_spec(module) is not None


def import_optional_module(module: str, extra: str) -> ModuleType:
    """
    Check if the required module is installed and load it
    """
    assert_optional_module_installed(module=module, extra=extra)
    return importlib.import_module(module)


def assert_optional_module_installed(module: str, extra):
    """
    Check if the required module is installed, or raise a human readable error message with instructions if it doesn't.
    """
    # Check if the module is installed
    if module_installed(module):
        return

    # Get the dependencies for the given extra
    try:
        dependencies = DEPENDENCIES[extra]
    except KeyError as ex:
        raise KeyError(f"Extra requirements '{extra}' is not defined.") from ex

    # Check if the module is part of the extra requirement
    if module not in dependencies:
        raise KeyError(f"Module '{module}' is not included in the extra requirements '{extra}'")

    # Atempt to guess the package manager
    if running_from_conda():
        cmd = "conda install "
    elif module_installed("pip"):
        cmd = "pip install "
    else:
        cmd = ""

    msg = (
        f"Missing optional module: `{module}`. Install it with `{cmd}{MAIN_PACKAGE}[{extra}]` "
        f"or `{cmd}{dependencies[module]}`"
    )

    # Raise the exception
    raise ModuleNotFoundError(msg)


def get_function(fn_name: str) -> Callable:
    """
    Get a function pointer by name
    """
    parts = fn_name.split(".")
    function_name = parts.pop()
    module_path = ".".join(parts) if parts else "builtins"
    try:
        module = import_module(module_path)
    except ModuleNotFoundError as ex:
        raise AttributeError(f"Module '{module_path}' does not exist (tried to resolve function '{fn_name}')!") from ex
    try:
        fn_ptr = getattr(module, function_name)
    except AttributeError as ex:
        raise AttributeError(f"Function '{function_name}' does not exist in module '{module_path}'!") from ex
    return fn_ptr
