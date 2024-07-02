# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
These functions can be used in the mapping files to apply filter functions to vision data
"""

from typing import List, Union

import pandas as pd

from power_grid_model_io.functions import has_value


def exclude_empty(row: pd.Series, col: str) -> bool:
    """
    filter out empty
    """
    if col not in row:
        raise ValueError(f"The column: '{col}' cannot be found for the filter")
    result = has_value(row[col])
    if isinstance(result, pd.Series):
        return result.item()
    return result


def exclude_value(row: pd.Series, col: str, value: Union[float, str]) -> bool:
    """
    filter out by match value
    """
    if col not in row:
        raise ValueError(f"The column: '{col}' cannot be found for the filter")
    result = row[col] != value
    if isinstance(result, pd.Series):
        return result.item()
    return result


def exclude_all_columns_empty_or_zero(row: pd.Series, cols: List[str]) -> bool:
    """
    filter out empty or zero values in multiple columns.
    This is same as not all(not exclude_value or not exclude_empty)
    """
    return any(exclude_value(row, col, 0) and exclude_empty(row, col) for col in cols)
