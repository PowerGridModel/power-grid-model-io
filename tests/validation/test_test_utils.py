# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
from power_grid_model import power_grid_meta_data

from .utils import component_attributes, select_values

MOCK_JSON_DATA = '{"node":[{"id": 0, "foo": 0},{"id": 0, "bar": 0}], "line": [{"id": 0, "baz": 0}]}'


@patch("pathlib.Path.open", mock_open(read_data=MOCK_JSON_DATA))
def test_component_attributes():
    # Act
    generator = component_attributes(Path("test.json"))

    # Assert
    assert list(generator) == [("line", "baz"), ("line", "id"), ("node", "bar"), ("node", "foo"), ("node", "id")]


def test_select_values():
    # Arrange
    input_node_dtype = power_grid_meta_data["input"]["node"]["dtype"]
    actual = {"node": np.array([(2, 2.0), (4, np.nan), (3, 3.0), (1, 1.0)], dtype=input_node_dtype)}
    expected = {"node": np.array([(4, 4.0), (1, np.nan), (3, 3.0)], dtype=input_node_dtype)}

    # Act
    actual_values, expected_values = select_values(
        actual=actual, expected=expected, component="node", attribute="u_rated"
    )

    # Assert
    pd.testing.assert_series_equal(actual_values, pd.Series([3.0, np.nan], index=[3, 4]))
    pd.testing.assert_series_equal(expected_values, pd.Series([3.0, 4.0], index=[3, 4]))
