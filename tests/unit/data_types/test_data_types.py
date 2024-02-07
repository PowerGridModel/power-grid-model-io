# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from pydantic import TypeAdapter

from power_grid_model_io.data_types import ExtraInfo, StructuredData


def test_extra_info():
    extra_info = {1: {"a": 123, "b": 1.23}, 2: {"c": (1.2, 3.4, 5.6), "d": "foo"}}

    # Expect no exception
    adapter = TypeAdapter(ExtraInfo)
    adapter.validate_python(extra_info)


def test_structured_data__single():
    data = {"node": [{"id": 1}, {"id": 2}], "line": [{"id": 3, "node_from": 1, "node_to": 2}]}

    # Expect no exception
    adapter = TypeAdapter(StructuredData)
    assert isinstance(adapter.validate_python(data), dict)


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
    adapter = TypeAdapter(StructuredData)
    assert isinstance(adapter.validate_python(data), list)
