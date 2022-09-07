# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
import importlib.util
import sys
from pathlib import Path
from typing import List, Optional

# extra_key -> module -> pip_package
DEPENDENCIES = {"tabular": {"yaml": "pyyaml"}, "excel": {"openpyxl": "openpyxl"}, "cli": {"typer": "typer[all]"}}


def running_from_conda_env() -> bool:
    return Path(sys.prefix, "conda-meta").exists()


def module_loaded(module: str) -> bool:
    return hasattr(sys, module)


def module_installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def import_optional_module(extra: str, module: str):
    assert_dependencies(extra=extra, modules=[module])
    if module_loaded(module):
        return getattr(sys, module)
    else:
        return importlib.import_module(module)


def assert_dependencies(extra: str, modules: Optional[List[str]] = None):
    # Get the dependencies for the given extra
    try:
        dependencies = DEPENDENCIES[extra]
    except KeyError:
        raise KeyError(f"Extra requirements '{extra}' is not defined.")

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
    module_name = __name__.split(".")[0]

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
