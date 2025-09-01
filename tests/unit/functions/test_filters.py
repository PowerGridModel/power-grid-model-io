# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
from typing import Any, Tuple
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from power_grid_model_io.functions.filters import exclude_all_columns_empty_or_zero, exclude_empty, exclude_value


@pytest.mark.parametrize(
    ("row_value", "expected_call"),
    [
        (1, 1),
        (pd.Series(1), 1),
    ],
)
@patch("power_grid_model_io.functions.filters.has_value")
def test_exclude_empty(mock_has_value: MagicMock, row_value: Any, expected_call: Any):
    col = "foo"
    row = pd.Series([row_value, "xyz"], index=["foo", "bar"])
    actual = exclude_empty(row=row, col=col)

    mock_has_value.assert_called_once_with(expected_call)
    assert actual is mock_has_value.return_value


def test_exclude_empty__invalid_col():
    row = pd.Series({"foo": 1})
    with pytest.raises(ValueError, match="The column: 'bar' cannot be found for the filter"):
        exclude_empty(row=row, col="bar")


@pytest.mark.parametrize(
    ("row_value", "check_value", "expected"),
    [
        (4.0, "x", True),
        (3.0, 3.0, False),
        (3.2, 3.1, True),
        (pd.Series([3]), 3, False),
        (pd.Series([3]), 2, True),
    ],
)
def test_exclude_value(row_value: float | pd.Series, check_value: float, expected: bool):
    row = pd.Series({"foo": row_value})
    actual = exclude_value(row=row, col="foo", value=check_value)
    assert actual == expected


def test_exclude_value__invalid_col():
    row = pd.Series({"foo": 1})
    with pytest.raises(ValueError, match="The column: 'bar' cannot be found for the filter"):
        exclude_value(row=row, col="bar", value=2)


@pytest.mark.parametrize(
    ("row_value", "expected"),
    [
        ((1.0, 2.0), True),
        ((1.0, 0.0), True),
        ((0.0, 1.0), True),
        ((0.0, 0.0), False),
        ((1.0, 3.0), True),
        (("", 1.0), True),
        (("", 0.0), False),
        (("", ""), False),
    ],
)
def test_exclude_all_columns_empty_or_zero(row_value: Tuple[float, float], expected: bool):
    row = pd.Series({"foo": row_value[0], "bar": row_value[1]})
    actual = exclude_all_columns_empty_or_zero(row=row, cols=["foo", "bar"])
    assert actual == expected
