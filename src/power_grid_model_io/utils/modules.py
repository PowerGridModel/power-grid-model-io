# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
Module utilities, expecially useful for loading optional dependencies
"""

from collections.abc import Callable, Mapping
from types import MappingProxyType

import power_grid_model_io.functions._functions as _fn
import power_grid_model_io.functions.filters as _filters
import power_grid_model_io.functions.phase_to_phase as _p2p

# Explicit allowlist: only these functions may be referenced by name in mapping files.
_ALLOWED_FUNCTIONS: Mapping[str, Callable] = MappingProxyType(
    {
        "power_grid_model_io.functions.has_value": _fn.has_value,
        "power_grid_model_io.functions.value_or_default": _fn.value_or_default,
        "power_grid_model_io.functions.value_or_zero": _fn.value_or_zero,
        "power_grid_model_io.functions.complex_inverse_real_part": _fn.complex_inverse_real_part,
        "power_grid_model_io.functions.complex_inverse_imaginary_part": _fn.complex_inverse_imaginary_part,
        "power_grid_model_io.functions.get_winding": _fn.get_winding,
        "power_grid_model_io.functions.degrees_to_clock": _fn.degrees_to_clock,
        "power_grid_model_io.functions.is_greater_than": _fn.is_greater_than,
        "power_grid_model_io.functions.both_zeros_to_nan": _fn.both_zeros_to_nan,
        "power_grid_model_io.functions.filters.exclude_empty": _filters.exclude_empty,
        "power_grid_model_io.functions.filters.exclude_value": _filters.exclude_value,
        "power_grid_model_io.functions.filters.exclude_all_columns_empty_or_zero": _filters.exclude_all_columns_empty_or_zero,  # noqa: E501
        "power_grid_model_io.functions.phase_to_phase.relative_no_load_current": _p2p.relative_no_load_current,
        "power_grid_model_io.functions.phase_to_phase.reactive_power": _p2p.reactive_power,
        "power_grid_model_io.functions.phase_to_phase.power_wind_speed": _p2p.power_wind_speed,
        "power_grid_model_io.functions.phase_to_phase.get_winding_from": _p2p.get_winding_from,
        "power_grid_model_io.functions.phase_to_phase.get_winding_to": _p2p.get_winding_to,
        "power_grid_model_io.functions.phase_to_phase.get_winding_1": _p2p.get_winding_1,
        "power_grid_model_io.functions.phase_to_phase.get_winding_2": _p2p.get_winding_2,
        "power_grid_model_io.functions.phase_to_phase.get_winding_3": _p2p.get_winding_3,
        "power_grid_model_io.functions.phase_to_phase.get_clock": _p2p.get_clock,
        "power_grid_model_io.functions.phase_to_phase.get_clock_12": _p2p.get_clock_12,
        "power_grid_model_io.functions.phase_to_phase.get_clock_13": _p2p.get_clock_13,
        "power_grid_model_io.functions.phase_to_phase.reactive_power_to_susceptance": _p2p.reactive_power_to_susceptance,  # noqa: E501
        "power_grid_model_io.functions.phase_to_phase.pvs_power_adjustment": _p2p.pvs_power_adjustment,
    }
)


def get_function(fn_name: str) -> Callable:
    try:
        return _ALLOWED_FUNCTIONS[fn_name]
    except KeyError as ex:
        raise AttributeError(f"'{fn_name}' is not an allowed mapping function") from ex
