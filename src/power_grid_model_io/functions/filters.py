# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
These functions can be used in the mapping files to apply filter functions to vision data
"""

from typing import List

import pandas as pd

from power_grid_model_io.functions import has_value


def filter_empty(row: pd.Series, col: str) -> bool:
    """
    filter out empty
    """
    return has_value(row[col]).values[0]


def filter_by_value(row: pd.Series, col: str, value: float | str) -> bool:
    """
    filter out by match value
    """
    return (row[col] != value).values[0]


def filter_all_columns_empty_or_zero(row: pd.Series, cols: List[str]) -> bool:
    """
    filter out empty or zero values in multiple columns
    """
    return any(filter_by_value(row, col, 0) & filter_empty(row, col) for col in cols)
