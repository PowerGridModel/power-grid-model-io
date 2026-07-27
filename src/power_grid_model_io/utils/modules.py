# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
Module utilities, expecially useful for loading optional dependencies
"""

import inspect
from collections.abc import Callable
from importlib import import_module


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


# Only functions living inside these (sub)modules can ever be resolved from a mapping file.
ALLOWED_MODULES = [
    "power_grid_model_io.functions",
]

# we can add public functions that list what are the allowed functions in mapping files in general and in cel expressions
_ALLOWED_FUNCTIONS: dict[str, Callable] = {}
_ALLOWED_CEL_FUNCTIONS: dict[str, Callable] = {}


def allowed_in_mapping(name: str, cel: bool = False) -> Callable:
    """Register a function under an explicit dotted name usable in mapping files."""

    def decorator(fn):
        _ALLOWED_FUNCTIONS[name] = fn
        if cel:
            _ALLOWED_CEL_FUNCTIONS[name] = fn
        return fn

    return decorator


def get_allowed_function_strict(fn_name: str) -> Callable:
    try:
        return _ALLOWED_FUNCTIONS[fn_name]
    except KeyError as ex:
        raise AttributeError(f"'{fn_name}' is not an allowed mapping function") from ex


# leaving this commented out as it's a different -less restrictive- approach
# def get_allowed_function(fn_name: str) -> Callable:
#     """
#     Like get_function, but only resolves names within an explicit allow-list of trusted modules, and only
#     returns plain functions - never modules, classes, or other objects that could be misused.
#     """
#     parts = fn_name.split(".")
#     function_name = parts.pop()
#     module_path = ".".join(parts) if parts else "builtins"

#     if not any(module_path == allowed or module_path.startswith(allowed + ".") for allowed in ALLOWED_MODULES):
#         raise AttributeError(
#             f"Module '{module_path}' is not in the allow-list of trusted modules for mapping functions "
#             f"(tried to resolve '{fn_name}'). Allowed: {ALLOWED_MODULES}"
#         )

#     try:
#         module = import_module(module_path)
#     except ModuleNotFoundError as ex:
#         raise AttributeError(f"Module '{module_path}' does not exist (tried to resolve function '{fn_name}')!") from ex

#     try:
#         fn_ptr = getattr(module, function_name)
#     except AttributeError as ex:
#         raise AttributeError(f"Function '{function_name}' does not exist in module '{module_path}'!") from ex

#     if not inspect.isfunction(fn_ptr):
#         raise AttributeError(
#             f"'{fn_name}' resolved to a {type(fn_ptr).__name__}, not a function; only plain functions are allowed"
#         )

#     return fn_ptr
