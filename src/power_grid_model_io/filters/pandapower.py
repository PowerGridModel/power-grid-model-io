# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
These functions can be used in the mapping files to apply functions to pandapower data
"""
import re
from typing import Tuple

from power_grid_model import WindingType

from power_grid_model_io.filters import get_winding

CONNECTION_PATTERN_PP = re.compile(r"(Y|YN|D|Z|ZN)(y|yn|d|z|zn)\d*")
CONNECTION_PATTERN_PP_3WDG = re.compile(r"(Y|YN|D|Z|ZN)(y|yn|d|z|zn)(y|yn|d|z|zn)\d*")


def _split_string(value: str) -> Tuple[str, str]:
    """
    Split the string of vector_group from PP into winding_from and winding_to of PGM
    """
    match = CONNECTION_PATTERN_PP.fullmatch(value)
    if not match:
        raise ValueError(f"Invalid transformer connection string: '{value}'")
    return match.group(1), match.group(2)


def get_transformer_winding_from(vector_group: str) -> WindingType:
    """
    Extract winding_from of a transformer
    """
    the_tuple = _split_string(vector_group)
    return get_winding(the_tuple[0], neutral_grounding=True)


def get_transformer_winding_to(vector_group: str) -> WindingType:
    """
    Extract winding_to of a transformer
    """
    the_tuple = _split_string(vector_group)
    return get_winding(the_tuple[1], neutral_grounding=True)


def _split_string_3wdg(value: str) -> Tuple[str, str, str]:
    """
    Split the string of vector_group from PP into winding_1, winding_2 and winding_3 of PGM
    """
    match = CONNECTION_PATTERN_PP_3WDG.fullmatch(value)
    if not match:
        raise ValueError(f"Invalid transformer connection string: '{value}'")
    return match.group(1), match.group(2), match.group(3)


def get_3wdgtransformer_winding_1(vector_group: str) -> WindingType:
    """
    Extract winding_1 of a three winding transformer
    """
    the_tuple = _split_string_3wdg(vector_group)
    return get_winding(the_tuple[0], neutral_grounding=True)


def get_3wdgtransformer_winding_2(vector_group: str) -> WindingType:
    """
    Extract winding_2 of a three winding transformer
    """
    the_tuple = _split_string_3wdg(vector_group)
    return get_winding(the_tuple[1], neutral_grounding=True)


def get_3wdgtransformer_winding_3(vector_group: str) -> WindingType:
    """
    Extract winding_3 of a three winding transformer
    """
    the_tuple = _split_string_3wdg(vector_group)
    return get_winding(the_tuple[2], neutral_grounding=True)
