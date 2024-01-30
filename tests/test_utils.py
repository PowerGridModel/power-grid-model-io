# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock, call, patch

import numpy as np
import pytest
import structlog
from structlog.testing import capture_logs

from .utils import MockDf, MockVal, assert_log_exists, assert_log_match, assert_struct_array_equal, contains, idx_to_str


@pytest.fixture
def captured_logs():
    with capture_logs() as captured:
        structlog.get_logger().info("Test info message", foo=123)
        structlog.get_logger().debug("Test debug message", bar=456)
    return captured


def test_dict_in_dict():
    # Act / Assert
    assert contains({}, {})
    assert contains({"a": 1}, {"a": 1})
    assert contains({"a": 1}, {"a": 1, "b": 2})
    assert contains({"a": 1, "b": 2}, {"a": 1, "b": 2})
    assert contains({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3})
    assert not contains({"a": 1}, {})
    assert not contains({"a": 1}, {"a": 2})
    assert not contains({"a": 1}, {"b": 1})
    assert not contains({"a": 1}, {"b": 2})
    assert not contains({"a": 1, "b": 2}, {"c": 3})


@patch("pandas.DataFrame")
@patch("pandas.testing.assert_frame_equal")
def test_assert_struct_array_equal(mock_assert_frame_equal: MagicMock, mock_data_frame: MagicMock):
    # Arrange
    actual_np = MagicMock()
    actual_pd = MagicMock()
    expected_np = MagicMock()
    expected_pd = MagicMock()
    mock_data_frame.side_effect = [actual_pd, expected_pd]

    # Act
    assert_struct_array_equal(actual=actual_np, expected=expected_np)

    # Assert
    assert mock_data_frame.call_args_list[0] == call(actual_np)
    assert mock_data_frame.call_args_list[1] == call(expected_np)
    mock_assert_frame_equal.assert_called_once_with(actual_pd, expected_pd, check_dtype=False)


def test_assert_struct_array_equal__dict_list():
    # Arrange
    actual = np.array([(12, 1, 2), (23, 2, 3)], dtype=[("id", int), ("from", int), ("to", int)])
    expected = [{"id": 12, "from": 1, "to": 2}, {"id": 23, "from": 2, "to": 3}]

    # Act / Assert
    assert_struct_array_equal(actual=actual, expected=expected)


def test_assert_log_exists(captured_logs):
    # Act / Assert
    assert_log_exists(captured_logs)
    assert_log_exists(captured_logs, "info")
    assert_log_exists(captured_logs, None, "Test debug message")
    assert_log_exists(captured_logs, bar=456)

    with pytest.raises(KeyError):
        assert_log_exists([])
    with pytest.raises(KeyError, match=r"{'log_level': 'error'}"):
        assert_log_exists(captured_logs, "error")
    with pytest.raises(KeyError, match=r"{'event': 'Test error message'}"):
        assert_log_exists(captured_logs, None, "Test error message")
    with pytest.raises(KeyError, match=r"{'foo': 456}"):
        assert_log_exists(captured_logs, foo=456)


def test_assert_log_exists__print(capsys, captured_logs):
    # Act
    with pytest.raises(KeyError):
        assert_log_exists(captured_logs, "error")
    stderr = capsys.readouterr().err

    # Assert
    assert "[info] Test info message {'foo': 123}" in stderr
    assert "[debug] Test debug message {'bar': 456}" in stderr


def test_assert_log_match(captured_logs):
    # Act / Assert
    assert_log_match(captured_logs[0])
    assert_log_match(captured_logs[0], "info")
    assert_log_match(captured_logs[1], None, "Test debug message")
    assert_log_match(captured_logs[1], bar=456)

    with pytest.raises(KeyError, match=r"{'log_level': 'error'}"):
        assert_log_match(captured_logs[0], "error")
    with pytest.raises(KeyError, match=r"{'event': 'Test error message'}"):
        assert_log_match(captured_logs[0], None, "Test error message")
    with pytest.raises(KeyError, match=r"{'foo': 456}"):
        assert_log_match(captured_logs[0], foo=456)


def test_idx_to_str():
    # Arrange
    class IndexAsString:
        def __getitem__(self, item):
            return idx_to_str(item)

    idx = IndexAsString()

    # Act / Assert
    assert idx[123] == "123"
    assert idx[123, 456] == "123, 456"
    assert idx["abc"] == "'abc'"
    assert idx[:] == ":"
    assert idx[1:] == "1:"
    assert idx[:2] == ":2"
    assert idx[1:2] == "1:2"
    assert idx[::3] == "::3"
    assert idx[1::3] == "1::3"
    assert idx[:2:3] == ":2:3"
    assert idx[1:2:3] == "1:2:3"
    assert idx[::-1] == "::-1"
    assert idx[:-1] == ":-1"
    assert idx[-2:-1] == "-2:-1"
    assert idx[1:2:3, 4:5:6] == "1:2:3, 4:5:6"


def test_mock_fn__operators():
    # Arrange
    x = MockVal("x")
    y = MockVal("y")
    z = MockVal("z")

    # Act / Assert
    assert x + y == y + x
    assert (x + y) + z == x + (y + z)
    assert x - (y + z) == x - y - z


def test_mock_val():
    assert callable(MockVal)  # TODO


def test_mock_df():
    assert callable(MockDf)  # TODO
