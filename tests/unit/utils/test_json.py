# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import io

import numpy as np

from power_grid_model_io.utils.json import compact_json_dump


def test_compact_json_dump():
    data = {
        "node": [{"id": 1, "x": 2}, {"id": 3, "x": 4}],
        "line": [{"id": 5, "x": 6}, {"id": 7, "x": {"y": 8.1, "z": 8.2}}],
    }

    string_stream = io.StringIO()
    compact_json_dump(data, string_stream, indent=2, max_level=0)
    assert (
        string_stream.getvalue()
        == """{"node": [{"id": 1, "x": 2}, {"id": 3, "x": 4}], "line": [{"id": 5, "x": 6}, {"id": 7, "x": {"y": 8.1, "z": 8.2}}]}"""
    )

    string_stream = io.StringIO()
    compact_json_dump(data, string_stream, indent=2, max_level=1)
    assert (
        string_stream.getvalue()
        == """{
  "node": [{"id": 1, "x": 2}, {"id": 3, "x": 4}],
  "line": [{"id": 5, "x": 6}, {"id": 7, "x": {"y": 8.1, "z": 8.2}}]
}
"""
    )

    string_stream = io.StringIO()
    compact_json_dump(data, string_stream, indent=2, max_level=2)
    assert (
        string_stream.getvalue()
        == """{
  "node":
    [{"id": 1, "x": 2}, {"id": 3, "x": 4}],
  "line":
    [{"id": 5, "x": 6}, {"id": 7, "x": {"y": 8.1, "z": 8.2}}]
}
"""
    )

    string_stream = io.StringIO()
    compact_json_dump(data, string_stream, indent=2, max_level=3)
    assert (
        string_stream.getvalue()
        == """{
  "node":
    [
      {"id": 1, "x": 2},
      {"id": 3, "x": 4}
    ],
  "line":
    [
      {"id": 5, "x": 6},
      {"id": 7, "x": {"y": 8.1, "z": 8.2}}
    ]
}
"""
    )


def test_compact_json_dump_string():
    data = "test"

    string_stream = io.StringIO()
    compact_json_dump(data, string_stream, indent=2, max_level=2)
    assert string_stream.getvalue() == '"test"'


def test_compact_json_dump_deep():
    data = {
        "foo": 1,
        "bar": {"x": 2, "y": 3},
    }

    string_stream = io.StringIO()
    compact_json_dump(data, string_stream, indent=2, max_level=10)
    assert (
        string_stream.getvalue()
        == """{
  "foo": 1,
  "bar":
    {
      "x": 2,
      "y": 3
    }

}
"""
    )


def test_compact_json_dump_batch():
    data = [
        {
            "node": [{"id": 1, "x": 2}, {"id": 3, "x": 4}],
            "line": [{"id": 5, "x": 6}, {"id": 7, "x": {"y": 8.1, "z": 8.2}}],
        },
        {
            "line": [{"id": 9, "x": 10}, {"id": 11, "x": 12}],
        },
    ]
    string_stream = io.StringIO()
    compact_json_dump(data, string_stream, indent=2, max_level=4)
    assert (
        string_stream.getvalue()
        == """[
  {
    "node":
      [
        {"id": 1, "x": 2},
        {"id": 3, "x": 4}
      ],
    "line":
      [
        {"id": 5, "x": 6},
        {"id": 7, "x": {"y": 8.1, "z": 8.2}}
      ]
  }
,
  {
    "line":
      [
        {"id": 9, "x": 10},
        {"id": 11, "x": 12}
      ]
  }

]"""
    )
