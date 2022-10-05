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
from typing import Callable, List, Optional

# extra_key -> module -> pip_package
DEPENDENCIES = {
    "cli": {"typer": "typer[all]"},
}


def running_from_conda_env() -> bool:
    """
    Check if the conda is used
    """
    return Path(sys.prefix, "conda-meta").exists()


def module_loaded(module: str) -> bool:
    """
    Check if the module is already loaded
    """
    return hasattr(sys, module)


def module_installed(module: str) -> bool:
    """
    Check if the module is installed
    """
    return importlib.util.find_spec(module) is not None


def import_optional_module(module: str, extra: str):
    """
    Check if the required module is installed and load it
    """
    assert_dependencies(extra=extra, modules=[module])
    if module_loaded(module):
        return getattr(sys, module)
    return importlib.import_module(module)


def assert_dependencies(extra: str, modules: Optional[List[str]] = None):
    """
    Check if the required module is installed, or raise a human readable errormessage with instructions if it doesn't.
    """
    # Get the dependencies for the given extra
    try:
        dependencies = DEPENDENCIES[extra]
    except KeyError as ex:
        raise KeyError(f"Extra requirements '{extra}' is not defined.") from ex

    # Get, or validate the modules
    if modules is not None:
        for module in modules:
            if module not in dependencies:
                raise KeyError(f"Module '{module} is not included in the extra requirements '{extra}'")
    else:
        modules = list(dependencies.keys())

    # Check which modules are missing
    missing = [module for module in modules if not module_loaded(module) and not module_installed(module)]
    if not missing:
        return

    # Define the main module name
    module_name = __name__.split(".", maxsplit=1)[0]

    # Atempt to guess the package manager
    if running_from_conda_env():
        cmd = "conda install "
    elif module_installed("pip"):
        cmd = "pip install "
    else:
        cmd = ""

    # Are we missing just one module, or multiple?
    if len(missing) == 1:
        msg = (
            f"Missing optional module: `{missing[0]}`. Install it with `{cmd}{module_name}[{extra}]` "
            f"or `{cmd}{dependencies[missing[0]]}`"
        )
    else:
        msg = (
            f"Missing optional modules: {', '.join(missing)}. Install them with `{cmd}{module_name}[{extra}]` "
            f"or `{cmd}{' '.join(dependencies[m] for m in missing)}`"
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
        raise AttributeError(f"Function: {fn_name} does not exist") from ex
    try:
        fn_ptr = getattr(module, function_name)
    except AttributeError as ex:
        raise AttributeError(f"Function: {function_name} does not exist in {module_path}") from ex
    return fn_ptr
