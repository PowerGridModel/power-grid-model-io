# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pydantic import parse_obj_as

from power_grid_model_io.data_types import ExtraInfo, ExtraInfoLookup, StructuredData


def test_extra_info():
    extra_info = {
        "a": 123,  # NominalValue = int
        "b": 1.23,  # RealValue = float
        "c": (1.2, 3.4, 5.6),  # AsymValue = Tuple[float, float, float]
        "d": "foo",  # str
    }

    # Expect no exception
    parse_obj_as(ExtraInfo, extra_info)


def test_extra_info_lookup():
    extra_info_lookup = {1: {"a": 123, "b": 1.23}, 2: {"c": (1.2, 3.4, 5.6), "d": "foo"}}

    # Expect no exception
    parse_obj_as(ExtraInfoLookup, extra_info_lookup)


def test_structured_data__single():
    data = {"node": [{"id": 1}, {"id": 2}], "line": [{"id": 3, "node_from": 1, "node_to": 2}]}

    # Expect no exception
    assert isinstance(parse_obj_as(StructuredData, data), dict)


def test_structured_data__batch():
    data = [
        {
            "load": [
                {"id": 1, "p": 111.0},
                {"id": 2, "p": 222.0},
            ]
        },
        {
            "load": [
                {"id": 1, "p": 333.0},
                {"id": 2, "p": 444.0},
            ]
        },
    ]

    # Expect no exception
    assert isinstance(parse_obj_as(StructuredData, data), list)
