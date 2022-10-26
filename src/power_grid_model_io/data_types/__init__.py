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
ExtraInfoLookup = Dict[int, ExtraInfo]
StructuredData = Union[Dict[str, List[Dict[str, Any]]], List[Dict[str, List[Dict[str, Any]]]]]
