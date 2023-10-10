# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
General regular expressions
"""

import re
from typing import Dict, Optional

_TRAFO_CONNECTION_RE = re.compile(r"^(Y|YN|D|Z|ZN)(y|yn|d|z|zn)(\d|1[0-2])?$")


def parse_trafo_connection(string: str) -> Optional[Dict[str, str]]:
    r"""Parse a trafo connection string if possible.

    Matches the following regular expression to the winding_from and winding_to codes.
    Optionally checks the clock number:

    ^               Start of the string
    (Y|YN|D|Z|ZN)   From winding type
    (y|yn|d|z|zn)   To winding type
    (\d|1[0-2])?    Optional clock number (0-12)
    $               End of the string

    Args:
        string (str): The input string.

    Returns:
        Optional[Dict[str, str]]: The parameters of the trafo connection if correct, else None.
    """
    match = _TRAFO_CONNECTION_RE.fullmatch(string)
    if not match:
        return None

    return {"winding_from": match.group(1), "winding_to": match.group(2), "clock_number": match.group(3)}


_TRAFO3_CONNECTION_RE = re.compile(r"^(Y|YN|D|Z|ZN)(y|yn|d|z|zn)(\d|1[0-2])?(y|yn|d|z|zn)(\d|1[0-2])?$")


def parse_trafo3_connection(string: str) -> Optional[Dict[str, str]]:
    r"""Parse a trafo connection string if possible.

    Matches the following regular expression to the winding_from and winding_to codes.
    Optionally checks the clock number:

    ^               Start of the string
    (Y|YN|D|Z|ZN)   First winding type
    (y|yn|d|z|zn)   Second winding type
    (\d|1[0-2])     Clock number (0-12)
    (y|yn|d|z|zn)   Third winding type
    (\d|1[0-2])     Clock number (0-12)
    $               End of the string

    Args:
        string (str): The input string.

    Returns:
        Optional[Dict[str, str]]: The parameters of the trafo connection if correct, else None.
    """
    match = _TRAFO3_CONNECTION_RE.fullmatch(string)
    if not match:
        return None

    return {
        "winding_1": match.group(1),
        "winding_2": match.group(2),
        "winding_3": match.group(4),
        "clock_12": match.group(3),
        "clock_13": match.group(5),
    }


def parse_node_ref(string: str) -> Optional[Dict[str, str]]:
    """Parse a node reference. string if possible.

    Matches if the input is the word 'node' with an optional prefix or suffix. E.g.:

    - node
    - from_node
    - node_1

    Args:
        string (str): The input string.

    Returns:
        Optional[Dict[str, str]]: The prefix and suffix if the input string is a reference to a node, else None.
    """
    if "node" not in string:
        return None

    prefix_and_suffix = string.split("node")
    if len(prefix_and_suffix) != 2:
        return None

    prefix, suffix = prefix_and_suffix
    if prefix != "" and not prefix.endswith("_"):
        return None
    if suffix != "" and not suffix.startswith("_"):
        return None

    return {"prefix": prefix, "suffix": suffix}


PVS_EFFICIENCY_TYPE_RE = re.compile(r"[ ,.]1 pu: (95|97) %")
r"""
Regular expressions to match the efficiency type percentage at 1 pu, eg:
    - 0,1 pu: 93 %; 1 pu: 97 %
    - 0,1..1 pu: 95 %
1 pu            After 1 pu  '1 pu:'
(95|97)         95 or 97 % type
%               before  '%'
"""
