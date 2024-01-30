# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
Common data types used in the Power Grid Model project
"""

from typing import Any, Dict, List, Union

ExtraInfo = Dict[int, Any]
"""
ExtraInfo is information about power grid model objects that are not part of the calculations. E.g. the original ID or
name of a node, or the material of a cable (line) etc.

It is a dictionary with numerical keys corresponding to the ids in input_data etc. The values are dictionaries with
textual keys. Their values may be anything, but it is advised to use only JSON serializable types like numerical values,
strings, lists, dictionaries etc.

    {
        1: {
            "length_km": 123.4,
            "material": "Aluminuminuminum",
        },
        2: {
            "id_reference": {
                "table": "load",
                "name": "const_power",
                "index": 101
            }
        }
    }
"""

ExtraInfoLookup = ExtraInfo
"""
Legacy type name; use ExtraInfo instead!
"""

StructuredData = Union[Dict[str, List[Dict[str, Any]]], List[Dict[str, List[Dict[str, Any]]]]]
"""
Structured data is a multi dimensional structure (component_type -> objects -> attribute -> value) or a list of those
dictionaries:
    {
      "node":
        [
          {"id": 0, "u_rated": 110000.0},
          {"id": 1, "u_rated": 110000.0},
        ],
      "line":
        [
          {"id": 2, "from_node": 0, "to_node": 1, "from_status": 1, "to_status": 1}
        ]
      "source":
        [
          {"id": 3, "node": 0, "status": 1, "u_ref": 1.0}
        ]
    }
"""
