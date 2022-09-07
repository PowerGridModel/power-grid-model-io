# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
import importlib.util
import sys
from pathlib import Path

# extra_key -> module -> pip_package
DEPENDENCIES = {"tabular": {"yaml": "pyyaml"}, "excel": {"openpyxl": "openpyxl"}, "cli": {"typer": "typer[all]"}}


def running_from_conda_env() -> bool:
    return Path(sys.prefix, "conda-meta").exists()


def module_installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


def assert_dependencies(extra: str):
    # Find missing modules, if any
    dependencies = DEPENDENCIES.get(extra, {})
    missing = [module for module in dependencies.keys() if not module_installed(module)]
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
