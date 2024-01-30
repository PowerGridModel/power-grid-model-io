# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import io
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from power_grid_model_io.utils.json import JsonEncoder, compact_json_dump


def test_compact_json_dump():
    data = {
        "node": [{"id": 1, "x": 2}, {"id": 3, "x": np.int64(4)}],
        "line": [{"id": 5, "x": 6}, {"id": 7, "x": {"y": 8.1, "z": np.float64(8.2)}}],
    }

    string_stream = io.StringIO()
    compact_json_dump(data, string_stream, indent=2, max_level=0)
    assert (
        string_stream.getvalue()
        == """{"node": [{"id": 1, "x": 2}, {"id": 3, "x": 4}],"""
        + """ "line": [{"id": 5, "x": 6}, {"id": 7, "x": {"y": 8.1, "z": 8.2}}]}"""
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
            "node": [{"id": 1, "x": 2}, {"id": 3, "x": np.int64(4)}],
            "line": [{"id": 5, "x": 6}, {"id": 7, "x": {"y": 8.1, "z": np.float64(8.2)}}],
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


@pytest.mark.parametrize(
    ("numpy_type", "native_type"),
    [
        (np.int8, int),
        (np.int16, int),
        (np.int32, int),
        (np.int64, int),
        (np.uint8, int),
        (np.uint16, int),
        (np.uint32, int),
        (np.uint64, int),
        (np.float16, float),
        (np.float32, float),
        (np.float64, float),
    ],
)
def test_json_encoder(numpy_type: type, native_type: type):
    # Arrange
    encoder = JsonEncoder()
    value = numpy_type(123)

    # Act
    value = encoder.default(value)

    # Assert
    assert type(value) == native_type
    assert value == 123


def test_json_encoder__np_array():
    # Arrange
    encoder = JsonEncoder()
    value = np.array([1, 2, 3])

    # Act
    value = encoder.default(value)

    # Assert
    assert type(value) == list
    assert value == [1, 2, 3]


@patch("power_grid_model_io.utils.json.json.JSONEncoder.default")
def test_json_encoder__super(mock_super: MagicMock):
    # Arrange
    encoder = JsonEncoder()
    value = "string value"

    # Act
    encoder.default(value)

    # Assert
    mock_super.assert_called_once_with(value)
