# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd
import pytest

from power_grid_model_io.data_types import TabularData
from power_grid_model_io.mappings.unit_mapping import UnitMapping
from power_grid_model_io.mappings.value_mapping import ValueMapping


@pytest.fixture()
def nodes() -> pd.DataFrame:
    return pd.DataFrame([(0, 150e3), (1, 10.5e3), (2, 400.0)], columns=["id", "u_rated"])


@pytest.fixture()
def nodes_iso() -> pd.DataFrame:
    return pd.DataFrame(
        [(0, 150e3), (1, 10.5e3), (2, 400.0)], columns=pd.MultiIndex.from_tuples((("id", None), ("u_rated", "V")))
    )


@pytest.fixture()
def nodes_kv() -> pd.DataFrame:
    return pd.DataFrame(
        [(0, 150), (1, 10.5), (2, 0.4)], columns=pd.MultiIndex.from_tuples((("id", None), ("u_rated", "kV")))
    )


@pytest.fixture()
def nodes_vl() -> pd.DataFrame:
    return pd.DataFrame([(0, "hv"), (1, "mv"), (2, "lv")], columns=("id", "u_rated"))


@pytest.fixture()
def lines() -> pd.DataFrame:
    return pd.DataFrame([(2, 0, 1), (3, 2, 3)], columns=("id", "from_node", "to_node"))


def test_tabular_data__get_column(nodes: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes, lines=lines)

    # Act
    col_data = data.get_column(table_name="nodes", column_name="u_rated")

    # Assert
    pd.testing.assert_series_equal(col_data, pd.Series([150e3, 10.5e3, 400.0], name="u_rated"))


def test_tabular_data__get_column__iso(nodes_iso: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes_iso, lines=lines)
    data.set_unit_multipliers(UnitMapping({"V": None}))

    # Act
    col_data = data.get_column(table_name="nodes", column_name="u_rated")

    # Assert
    pd.testing.assert_series_equal(col_data, pd.Series([150e3, 10.5e3, 400.0], name=("u_rated", "V")))


def test_tabular_data__get_column__iso_exception(nodes_iso: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes_iso, lines=lines)

    # Act / Assert
    with pytest.raises(KeyError, match=r"V.+u_rated.+nodes"):
        data.get_column(table_name="nodes", column_name="u_rated")


def test_tabular_data__get_column__unit(nodes_kv: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes_kv, lines=lines)
    data.set_unit_multipliers(UnitMapping({"V": {"kV": 1e3}}))

    # Act
    col_data = data.get_column(table_name="nodes", column_name="u_rated")

    # Assert
    pd.testing.assert_series_equal(col_data, pd.Series([150e3, 10.5e3, 400.0], name=("u_rated", "V")))


def test_tabular_data__get_column__unit_exception(nodes_kv: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes_kv, lines=lines)

    # Act / Assert
    with pytest.raises(KeyError, match=r"kV.+u_rated.+nodes"):
        data.get_column(table_name="nodes", column_name="u_rated")


def test_tabular_data__get_column__substitution(nodes_vl: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes_vl, lines=lines)
    data.set_substitutions(ValueMapping({"u_rated": {"mv": 10500.0, "lv": 400.0, "hv": 150000.0}}))

    # Act
    col_data = data.get_column(table_name="nodes", column_name="u_rated")

    # Assert
    pd.testing.assert_series_equal(col_data, pd.Series([150e3, 10.5e3, 400.0], name="u_rated"))


def test_tabular_data__get_column__substitution_exception(nodes_vl: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes_vl, lines=lines)
    data.set_substitutions(ValueMapping({"u_rated": {"MV": 10500.0, "LV": 400.0, "HV": 150000.0}}))

    # Act / Assert
    with pytest.raises(KeyError, match=r"hv.+u_rated.+nodes"):
        data.get_column(table_name="nodes", column_name="u_rated")


def test_tabular_data__contains(nodes: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes, lines=lines)

    # Act / Assert
    assert "nodes" in data
    assert "lines" in data


def test_tabular_data__keys(nodes: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes, lines=lines)

    # Act
    keys = list(data.keys())

    # Assert
    assert keys == ["nodes", "lines"]


def test_tabular_data__items(nodes: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes, lines=lines)

    # Act
    items = list(data.items())

    # Assert
    assert items == [("nodes", nodes), ("lines", lines)]
