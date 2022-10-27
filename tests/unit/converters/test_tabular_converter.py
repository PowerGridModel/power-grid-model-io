# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Any, Dict, Optional, Tuple, Union
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from power_grid_model_io.converters.tabular_converter import COL_REF_RE, NODE_REF_RE, TabularConverter
from power_grid_model_io.data_types.tabular_data import TabularData


def ref_cases():
    yield "OtherTable!ValueColumn[IdColumn=RefColumn]", (
        "OtherTable",
        "ValueColumn",
        None,
        None,
        "IdColumn",
        None,
        None,
        "RefColumn",
    )

    yield "OtherTable!ValueColumn[OtherTable!IdColumn=ThisTable!RefColumn]", (
        "OtherTable",
        "ValueColumn",
        "OtherTable!",
        "OtherTable",
        "IdColumn",
        "ThisTable!",
        "ThisTable",
        "RefColumn",
    )

    yield "OtherTable.ValueColumn[IdColumn=RefColumn]", None
    yield "ValueColumn[IdColumn=RefColumn]", None
    yield "OtherTable![IdColumn=RefColumn]", None


@pytest.mark.parametrize("value,groups", ref_cases())
def test_col_ref_pattern(value: str, groups: Optional[Tuple[Optional[str]]]):
    match = COL_REF_RE.fullmatch(value)
    if groups is None:
        assert match is None
    else:
        assert match is not None
        assert match.groups() == groups


def test_node_ref_pattern__pos():
    assert NODE_REF_RE.fullmatch("node")
    assert NODE_REF_RE.fullmatch("from_node")
    assert NODE_REF_RE.fullmatch("to_node")
    assert NODE_REF_RE.fullmatch("node_1")
    assert NODE_REF_RE.fullmatch("node_2")
    assert NODE_REF_RE.fullmatch("node_3")


def test_node_ref_pattern__neg():
    assert not NODE_REF_RE.fullmatch("nodes")
    assert not NODE_REF_RE.fullmatch("anode")
    assert not NODE_REF_RE.fullmatch("immunodeficient")


def test_apply_multiplier__no_multipliers():
    # Arrange
    converter = TabularConverter()
    data = pd.Series([-2.0, 0.0, 2.0])

    # Act
    result = converter._apply_multiplier(table="foo", column="bar", data=data)

    # Assert
    assert converter._multipliers is None
    pd.testing.assert_series_equal(data, pd.Series([-2.0, 0.0, 2.0]))
    pd.testing.assert_series_equal(result, pd.Series([-2.0, 0.0, 2.0]))


def test_apply_multiplier():
    # Arrange
    converter = TabularConverter()
    converter._multipliers = MagicMock()
    converter._multipliers.get_multiplier.return_value = 10.0
    data = pd.Series([-2.0, 0.0, 2.0])

    # Act
    result = converter._apply_multiplier(table="foo", column="bar", data=data)

    # Assert
    converter._multipliers.get_multiplier.assert_called_once_with(table="foo", attr="bar")
    pd.testing.assert_series_equal(data, pd.Series([-2.0, 0.0, 2.0]))
    pd.testing.assert_series_equal(result, pd.Series([-20.0, 0.0, 20.0]))


def test_apply_multiplier__no_attr_multiplier():
    # Arrange
    converter = TabularConverter()
    converter._multipliers = MagicMock()
    converter._multipliers.get_multiplier.side_effect = KeyError
    data = pd.Series([-2.0, 0.0, 2.0])

    # Act
    result = converter._apply_multiplier(table="foo", column="bar", data=data)

    # Assert
    converter._multipliers.get_multiplier.assert_called_once_with(table="foo", attr="bar")
    pd.testing.assert_series_equal(data, pd.Series([-2.0, 0.0, 2.0]))
    pd.testing.assert_series_equal(result, pd.Series([-2.0, 0.0, 2.0]))


@patch("power_grid_model_io.converters.tabular_converter.initialize_array")
@patch("power_grid_model_io.converters.tabular_converter.TabularConverter._convert_col_def_to_attribute")
def test__convert_table_to_component__condition(
    mock_convert_col_def_to_attribute: MagicMock, mock_initialize_array: MagicMock
):
    # Arrange
    converter = TabularConverter()
    data = TabularData(Nodes=pd.DataFrame([[1, 400.0], [1, 10.5e3], [2, 10.5e3], [3, 400.0]], columns=("id", "U")))
    attributes = {
        "id": "node_id",
        "u_rated": "U",
        "condition": {"power_grid_model_io.filters.is_greater_than": ["U", 1000.0]},
    }

    # Act
    converter._convert_table_to_component(
        data=data,
        data_type="input",
        table="Nodes",
        component="node",
        attributes=attributes,  # type: ignore
        extra_info=None,
    )

    # Assert
    mock_initialize_array.assert_called_once_with(data_type="input", component_type="node", shape=2)
    calls = mock_convert_col_def_to_attribute.call_args_list
    np.testing.assert_array_equal(calls[0][1]["selection"], np.array([False, True, True, False]))
    np.testing.assert_array_equal(calls[1][1]["selection"], np.array([False, True, True, False]))
