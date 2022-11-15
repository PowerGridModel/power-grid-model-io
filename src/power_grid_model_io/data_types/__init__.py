# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Common data types used in the Power Grid Model IO project
"""

from typing import Any, Dict, List, Union

from power_grid_model.data_types import AttributeValue

from power_grid_model_io.data_types.tabular_data import TabularData

ExtraInfo = Dict[str, Union[AttributeValue, str]]
"""
ExtraInfo is information about power grid model objects that are not part of the calculations. E.g. the original ID or
name of a node, or the material of a cable (line) etc. Extra info should be a dictionary with textual keys and
numerical or textual values:
    {
        "id": 123,
        "length_km": 123.4,
        "material": "Aluminuminuminum"
    }
"""

ExtraInfoLookup = Dict[int, ExtraInfo]
"""
An ExtraInfoLookup is a dictionary with numerical keys corresponding to the ids in input_data etc. The values are
ExtraInfo dictionaries.
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
