# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
General regular expressions
"""

import re
from typing import Dict

_TRAFO_CONNECTION_RE = re.compile(r"^(Y|YN|D|Z|ZN)(y|yn|d|z|zn)(\d|1[0-2])?$")


def parse_trafo_connection(string: str) -> Dict[str, str]:
    r"""Parse a trafo connection string.

    Matches the following regular expression to the winding_from and winding_to codes.
    Optionally checks the clock number:

    ^               Start of the string
    (Y|YN|D|Z|ZN)   From winding type
    (y|yn|d|z|zn)   To winding type
    (\d|1[0-2])?    Optional clock number (0-12)
    $               End of the string

    Args:
        string (str): The input string.

    Raises:
        ValueError: If the input is not a trafo connection string.

    Returns:
        Dict[str, str]: The parameters of the trafo connection.
    """
    match = _TRAFO_CONNECTION_RE.fullmatch(string)
    if not match:
        raise ValueError(f"Invalid transformer connection string: '{string}'")

    return {"winding_from": match.group(1), "winding_to": match.group(2), "clock": match.group(3)}


_TRAFO3_CONNECTION_RE = re.compile(r"^(Y|YN|D|Z|ZN)(y|yn|d|z|zn)(\d|1[0-2])?(y|yn|d|z|zn)(\d|1[0-2])?$")


def parse_trafo3_connection(string: str) -> Dict[str, str]:
    r"""Parse a trafo connection string.

    Matches the following regular expression to the winding_1, winding_2 and winding_3 codes.
    Optionally checks the clock numbers:

    ^               Start of the string
    (Y|YN|D|Z|ZN)   First winding type
    (y|yn|d|z|zn)   Second winding type
    (\d|1[0-2])     Clock number (0-12)
    (y|yn|d|z|zn)   Third winding type
    (\d|1[0-2])     Clock number (0-12)
    $               End of the string

    Args:
        string (str): The input string.

    Raises:
        ValueError: If the input is not a trafo connection string.

    Returns:
        Dict[str, str]: The parameters of the trafo connection.
    """
    match = _TRAFO3_CONNECTION_RE.fullmatch(string)
    if not match:
        raise ValueError(f"Invalid three winding transformer connection string: '{string}'")

    return {
        "winding_1": match.group(1),
        "winding_2": match.group(2),
        "winding_3": match.group(4),
        "clock_12": match.group(3),
        "clock_13": match.group(5),
    }


def parse_node_ref(string: str) -> Dict[str, str]:
    """Parse a node reference string.

    Matches if the input is the word 'node' with an optional prefix or suffix. E.g.:

    - node
    - from_node
    - node_1

    Args:
        string (str): The input string.

    Raises:
        ValueError: If the input string is not a node reference.

    Returns:
        Optional[Dict[str, str]]: The prefix and suffix (may be empty).
    """

    def _raise():
        raise ValueError(f"Invalid node reference string: '{string}'")

    if "node" not in string:
        _raise()

    prefix_and_suffix = string.split("node")
    if len(prefix_and_suffix) != 2:
        _raise()

    prefix, suffix = prefix_and_suffix
    if prefix and not prefix.endswith("_"):
        _raise()
    if suffix and not suffix.startswith("_"):
        _raise()

    return {"prefix": prefix, "suffix": suffix}


def is_node_ref(string: str) -> bool:
    """Return True if the string represents a node reference, else False.

    Like parse_node_ref, but without exceptions and result data.

    Args:
        string (str): The input string.

    Returns:
        bool: True if the string represents a node reference, else False.
    """
    try:
        parse_node_ref(string)
        return True
    except ValueError:
        return False


_PVS_EFFICIENCY_TYPE_RE = re.compile(r"[ ,.]1 pu: (95|97) %")


def parse_pvs_efficiency_type(string: str) -> str:
    r"""Parse a PVS efficiency type string.

    Matches the following regex to the efficiency type percentage at 1 pu.

    1 pu            After 1 pu  '1 pu:'
    (95|97)         95 or 97 % type
    %               before  '%'

    E.g.:

    - 0,1 pu: 93 %; 1 pu: 97 %
    - 0,1..1 pu: 95 %

    Args:
        string (str): The input string.

    Raises:
        ValueError: If the input string is not a PVS efficiency type.

    Returns:
        Optional[str]: The efficiency type percentage string at 1 pu.
    """
    match = _PVS_EFFICIENCY_TYPE_RE.search(string)
    if not match:
        raise ValueError(f"Invalid PVS efficiency type string: '{string}'")

    return match.group(1)
