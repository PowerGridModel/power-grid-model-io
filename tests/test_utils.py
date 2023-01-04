# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock, call, patch

from .utils import (
    MockDf,
    MockFn,
    MockVal,
    _dict_in_dict,
    assert_log_exists,
    assert_log_match,
    assert_struct_array_equal,
    idx_to_str,
)


def test_dict_in_dict():
    # Act / Assert
    assert _dict_in_dict({}, {})
    assert _dict_in_dict({"a": 1}, {"a": 1})
    assert _dict_in_dict({"a": 1}, {"a": 1, "b": 2})
    assert _dict_in_dict({"a": 1, "b": 2}, {"a": 1, "b": 2})
    assert _dict_in_dict({"a": 1, "b": 2}, {"a": 1, "b": 2, "c": 3})
    assert not _dict_in_dict({"a": 1}, {})
    assert not _dict_in_dict({"a": 1}, {"a": 2})
    assert not _dict_in_dict({"a": 1}, {"b": 1})
    assert not _dict_in_dict({"a": 1}, {"b": 2})
    assert not _dict_in_dict({"a": 1, "b": 2}, {"c": 3})


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
    mock_assert_frame_equal.assert_called_once_with(actual_pd, expected_pd)


def test_assert_log_exists():
    assert callable(assert_log_exists)  # TODO


def test_assert_log_match():
    assert callable(assert_log_match)  # TODO


def test_idx_to_str():
    assert callable(idx_to_str)  # TODO


def test_mock_fn():
    assert callable(MockFn)  # TODO


def test_mock_val():
    assert callable(MockVal)  # TODO


def test_mock_df():
    assert callable(MockDf)  # TODO
