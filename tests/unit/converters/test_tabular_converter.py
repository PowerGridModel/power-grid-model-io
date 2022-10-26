# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Optional, Tuple
from unittest.mock import MagicMock

import pandas as pd
import pytest

from power_grid_model_io.converters.tabular_converter import COL_REF_RE, NODE_REF_RE, TabularConverter


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
