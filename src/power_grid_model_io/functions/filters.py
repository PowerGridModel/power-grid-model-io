# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
These functions can be used in the mapping files to apply filter functions to vision data
"""

import pandas as pd

from power_grid_model_io.functions import has_value
from power_grid_model_io.utils.modules import allowed_in_mapping


def exclude_empty(row: pd.Series, col: str) -> bool:
    """
    filter out empty
    """
    if col not in row:
        raise ValueError(f"The column: '{col}' cannot be found for the filter")

    col_value = row[col]
    if isinstance(col_value, pd.Series):
        col_value = col_value.item()

    return has_value(col_value)


@allowed_in_mapping(
    "power_grid_model_io.functions.filters.exclude_value", cel=False
)  # needed because implementation is in private module _functions.py and python changes this in the __init__
# cel=False because this function isn't directly compatible as it expects a pd column
def exclude_value(row: pd.Series, col: str, value: float | str) -> bool:
    """
    filter out by match value
    """
    if col not in row:
        raise ValueError(f"The column: '{col}' cannot be found for the filter")

    col_value = row[col]
    if isinstance(col_value, pd.Series):
        col_value = col_value.item()

    return col_value != value


def exclude_all_columns_empty_or_zero(row: pd.Series, cols: list[str]) -> bool:
    """
    filter out empty or zero values in multiple columns.
    This is same as not all(not exclude_value or not exclude_empty)
    """
    return any(exclude_value(row, col, 0) and exclude_empty(row, col) for col in cols)
