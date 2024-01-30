# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import pandas as pd
from power_grid_model import power_grid_meta_data

from .utils import component_attributes, extract_extra_info, select_values

MOCK_JSON_DATA = '{"node":[{"id":0,"u_rated":0.0},{"id":0,"u_rated":0.0,"bar":0}],"line":[{"id":0,"i_n": 0}]}'


@patch("pathlib.Path.open", mock_open(read_data=MOCK_JSON_DATA))
def test_component_attributes():
    # Act
    generator = component_attributes(Path("test.json"), data_type="input")

    # Assert
    assert list(generator) == [("line", "i_n"), ("line", "id"), ("node", "id"), ("node", "u_rated")]


def test_select_values():
    # Arrange
    input_node_dtype = power_grid_meta_data["input"]["node"].dtype
    actual = {"node": np.array([(2, 2.0), (4, np.nan), (3, 3.0), (1, 1.0)], dtype=input_node_dtype)}
    expected = {"node": np.array([(4, 4.0), (1, np.nan), (3, 3.0)], dtype=input_node_dtype)}

    # Act
    actual_values, expected_values = select_values(
        actual=actual, expected=expected, component="node", attribute="u_rated"
    )

    # Assert
    pd.testing.assert_series_equal(actual_values, pd.Series([3.0, np.nan], index=np.array([3, 4], dtype=np.int32)))
    pd.testing.assert_series_equal(expected_values, pd.Series([3.0, 4.0], index=np.array([3, 4], dtype=np.int32)))


def test_extract_extra_info():
    # Arrange
    data = {
        "node": [{"id": 3, "name": "foo"}, {"id": 1, "u_rated": 400.0, "name": "bar"}],
        "line": [{"id": 2, "name": "baz", "color": "red"}, {"id": 4, "r1": 0.0, "c1": 0.0, "x1": 0.0}],
    }

    # Act
    extra_info = extract_extra_info(data=data, data_type="input")

    # Assert
    assert extra_info[1] == {"name": "bar"}
    assert extra_info[2] == {"name": "baz", "color": "red"}
    assert extra_info[3] == {"name": "foo"}
    assert 4 not in extra_info
    assert len(extra_info) == 3
