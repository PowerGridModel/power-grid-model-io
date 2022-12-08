# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Module utilities, expecially useful for loading optional dependencies
"""
from importlib import import_module
from typing import Callable


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
