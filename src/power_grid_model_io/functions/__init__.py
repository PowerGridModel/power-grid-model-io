# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

"""
These functions can be used in the mapping files to apply functions to tabular data
"""

from power_grid_model_io.functions._functions import (
    both_zeros_to_nan,
    complex_inverse_imaginary_part,
    complex_inverse_real_part,
    degrees_to_clock,
    get_winding,
    has_value,
    is_greater_than,
    value_or_default,
    value_or_zero,
)
