# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
General regular expressions
"""

import re
from typing import Dict, Optional

_TRAFO_CONNECTION_RE = re.compile(r"^(Y|YN|D|Z|ZN)(y|yn|d|z|zn)(\d|1[0-2])?$")


def get_trafo_connection(string: str) -> Optional[Dict[str, str]]:
    r"""Check whether the string is a trafo connection

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
        bool: Whether the input is a trafo connection.
    """
    match = _TRAFO_CONNECTION_RE.fullmatch(string)
    if not match:
        return None

    return {"winding_from": match.group(1), "winding_to": match.group(2), "clock_number": match.group(3)}


TRAFO3_CONNECTION_RE = re.compile(r"^(Y|YN|D|Z|ZN)(y|yn|d|z|zn)(\d|1[0-2])?(y|yn|d|z|zn)(\d|1[0-2])?$")
r"""
Regular expressions to the winding_1, winding_2 and winding_3 codes and optionally the clock number:
^               Start of the string
(Y|YN|D|Z|ZN)   First winding type
(y|yn|d|z|zn)   Second winding type
(\d|1[0-2])     Clock number (0-12)
(y|yn|d|z|zn)   Third winding type
(\d|1[0-2])     Clock number (0-12)
$               End of the string
"""


def is_node_ref(string: str) -> bool:
    """Check whether the string is a node reference.

    Returns True if and only if the input is the word node with an optional prefix or suffix, e.g.:

    - node
    - from_node
    - node_1

    Args:
        string (str): The input string.

    Returns:
        bool: Whether the input string is a reference to a node.
    """
    if "node" not in string:
        return False

    prefix_and_suffix = string.split("node")
    if len(prefix_and_suffix) != 2:
        return False

    prefix, suffix = prefix_and_suffix
    if prefix != "" and not prefix.endswith("_"):
        return False
    if suffix != "" and not suffix.startswith("_"):
        return False

    return True


PVS_EFFICIENCY_TYPE_RE = re.compile(r"[ ,.]1 pu: (95|97) %")
r"""
Regular expressions to match the efficiency type percentage at 1 pu, eg:
    - 0,1 pu: 93 %; 1 pu: 97 %
    - 0,1..1 pu: 95 %
1 pu            After 1 pu  '1 pu:'
(95|97)         95 or 97 % type
%               before  '%'
"""
