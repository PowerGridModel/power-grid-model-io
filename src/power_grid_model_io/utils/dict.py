# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
General dictionary utilities
"""
from copy import deepcopy
from typing import Dict


def merge_dicts(*dictionaries: Dict) -> Dict:
    """
    Merge two dictionaries, ignore duplicate key/values
    Args:
        *dictionaries: The dictionaries to be merges

    Returns: A (hard copied) combination of all dictionaries

    """
    if len(dictionaries) == 0:
        return {}

    result = deepcopy(dictionaries[0])
    for dictionary in dictionaries[1:]:
        for key, value in dictionary.items():
            if key not in result:
                result[key] = deepcopy(value)
            elif isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = merge_dicts(result[key], value)
            elif result[key] != value:
                raise KeyError(f"Clashing key '{key}' with different values in merge_dicts")
    return result
