# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
from structlog.testing import capture_logs

from power_grid_model_io.data_types import TabularData
from power_grid_model_io.mappings.unit_mapping import UnitMapping
from power_grid_model_io.mappings.value_mapping import ValueMapping

from ...utils import assert_log_exists


@pytest.fixture()
def nodes() -> pd.DataFrame:
    return pd.DataFrame([(0, 150e3), (1, 10.5e3), (2, 400.0)], columns=["id", "u_rated"])


@pytest.fixture()
def nodes_iso() -> pd.DataFrame:
    return pd.DataFrame(
        [(0, 150e3), (1, 10.5e3), (2, 400.0)], columns=pd.MultiIndex.from_tuples((("id", ""), ("u_rated", "V")))
    )


@pytest.fixture()
def nodes_kv() -> pd.DataFrame:
    return pd.DataFrame(
        [(0, 150), (1, 10.5), (2, 0.4)], columns=pd.MultiIndex.from_tuples((("id", ""), ("u_rated", "kV")))
    )


@pytest.fixture()
def nodes_vl() -> pd.DataFrame:
    return pd.DataFrame([(0, "hv"), (1, "mv"), (2, "lv")], columns=("id", "u_rated"))


@pytest.fixture()
def nodes_np() -> np.ndarray:
    return np.array([(0, 150e3), (1, 10.5e3), (2, 400.0)], dtype=[("id", "i4"), ("u_rated", "f4")])


@pytest.fixture()
def lines() -> pd.DataFrame:
    return pd.DataFrame([(2, 0, 1), (3, 2, 3)], columns=("id", "from_node", "to_node"))


def test_constructor__invalid_type():
    # Act / Assert
    with pytest.raises(TypeError, match=r"Invalid.*foo.*list"):
        TabularData(foo=[])  # type: ignore


def test_get_column(nodes: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes, lines=lines)

    # Act
    col_data = data.get_column(table_name="nodes", column_name="u_rated")

    # Assert
    pd.testing.assert_series_equal(col_data, pd.Series([150e3, 10.5e3, 400.0], name="u_rated"))


def test_get_column__numpy(nodes_np: np.ndarray):
    # Arrange
    data = TabularData(nodes=nodes_np)

    # Act
    col_data = data.get_column(table_name="nodes", column_name="u_rated")

    # Assert
    isinstance(col_data, pd.Series)
    pd.testing.assert_series_equal(col_data, pd.Series([150e3, 10.5e3, 400.0], name="u_rated", dtype=np.float32))


def test_get_column__numpy_is_a_reference(nodes_np: np.ndarray):
    # Arrange
    data = TabularData(nodes=nodes_np)

    # Act
    col_data = data.get_column(table_name="nodes", column_name="u_rated")
    col_data.loc[0] = 123.0  # << should update the source data

    # Assert
    np.testing.assert_array_equal(nodes_np["u_rated"], np.array([123.0, 10.5e3, 400.0]))


def test_get_column__iso(nodes_iso: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes_iso, lines=lines)
    data.set_unit_multipliers(UnitMapping({"V": None}))

    # Act
    col_data = data.get_column(table_name="nodes", column_name="u_rated")

    # Assert
    pd.testing.assert_series_equal(col_data, pd.Series([150e3, 10.5e3, 400.0], name=("u_rated", "V")))


def test_get_column__iso_exception(nodes_iso: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes_iso, lines=lines)

    # Act / Assert
    with pytest.raises(KeyError, match=r"V.+u_rated.+nodes"):
        data.get_column(table_name="nodes", column_name="u_rated")


def test_get_column__unit(nodes_kv: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes_kv, lines=lines)
    data.set_unit_multipliers(UnitMapping({"V": {"kV": 1e3}}))

    # Act
    col_data = data.get_column(table_name="nodes", column_name="u_rated")

    # Assert
    pd.testing.assert_series_equal(col_data, pd.Series([150e3, 10.5e3, 400.0], name=("u_rated", "V")))


def test_get_column__missing_unit(nodes_kv: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes_kv, lines=lines)

    # Act / Assert
    with pytest.raises(KeyError, match=r"kV.+u_rated.+nodes"):
        data.get_column(table_name="nodes", column_name="u_rated")


def test_get_column__invalid_multiplier(nodes_kv: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes_kv, lines=lines)
    data.set_unit_multipliers(UnitMapping({"V": {"kV": "1000x"}}))  # type: ignore

    # Act / Assert
    with pytest.raises(TypeError, match=r"1000x.*kV.*not numerical"):
        data.get_column(table_name="nodes", column_name="u_rated")


def test_get_column__non_numerical_value(lines: pd.DataFrame):
    # Arrange
    nodes_txt = pd.DataFrame(
        [(0, "150 kV"), (1, "10.5 kV"), (2, "0.4 kV")],
        columns=pd.MultiIndex.from_tuples([("id", ""), ("u_rated", "kV")]),
    )
    data = TabularData(nodes=nodes_txt, lines=lines)
    data.set_unit_multipliers(UnitMapping({"V": {"kV": 1e3}}))

    # Act / Assert
    with capture_logs() as cap_log:
        col_data = data.get_column(table_name="nodes", column_name="u_rated")
    pd.testing.assert_series_equal(col_data, pd.Series(["150 kV", "10.5 kV", "0.4 kV"], name=("u_rated", "")))
    assert_log_exists(cap_log, "error", "Failed to apply unit conversion; the column is not numerical.")


def test_get_column__no_unit_conversion(nodes_kv: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes_kv, lines=lines)

    # Act
    col_data = data.get_column(table_name="nodes", column_name="id")

    # Assert
    pd.testing.assert_series_equal(col_data, pd.Series([0, 1, 2], name="id"))


def test_get_column__substitution(nodes_vl: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes_vl, lines=lines)
    data.set_substitutions(ValueMapping({"u_rated": {"mv": 10500.0, "lv": 400.0, "hv": 150000.0}}))

    # Act
    col_data = data.get_column(table_name="nodes", column_name="u_rated")

    # Assert
    pd.testing.assert_series_equal(col_data, pd.Series([150e3, 10.5e3, 400.0], name="u_rated"))


def test_get_column__substitution__numpy(nodes_np: np.ndarray):
    # Arrange
    data = TabularData(nodes=nodes_np)
    data.set_substitutions(ValueMapping({"id": {0: 100, 1: 101, 2: 102}}))

    # Act
    col_data = data.get_column(table_name="nodes", column_name="id")

    # Assert
    pd.testing.assert_series_equal(col_data, pd.Series([100, 101, 102], name="id"))


def test_get_column__substitution_exception(nodes_vl: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes_vl, lines=lines)
    data.set_substitutions(ValueMapping({"u_rated": {"MV": 10500.0, "LV": 400.0, "HV": 150000.0}}))

    # Act / Assert
    with pytest.raises(KeyError, match=r"hv.+u_rated.+nodes"):
        data.get_column(table_name="nodes", column_name="u_rated")


def test_get_column__index(nodes: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes, lines=lines)

    # Act
    nodes_idx = data.get_column(table_name="nodes", column_name="index")
    lines_idx = data.get_column(table_name="lines", column_name="index")

    # Assert
    pd.testing.assert_series_equal(nodes_idx, pd.Series([0, 1, 2], name="index"))
    pd.testing.assert_series_equal(lines_idx, pd.Series([0, 1], name="index"))


@patch("power_grid_model_io.data_types.tabular_data.TabularData._apply_unit_conversion")
def test_get_column__sanity_check(mock_unit_conversion: MagicMock, nodes_iso: pd.DataFrame, nodes_kv: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes_kv)
    mock_unit_conversion.return_value = nodes_iso["u_rated"]

    # Act / Assert
    with pytest.raises(TypeError, match=r"u_rated.+unitless.+V"):
        data.get_column(table_name="nodes", column_name="u_rated")


def test_lazy_loading(nodes: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    def nodes_fn():
        return nodes

    def lines_fn():
        return lines

    data = TabularData(nodes=nodes_fn, lines=lines_fn)

    # Assert: No data loaded
    assert not isinstance(data._data["nodes"], pd.DataFrame)
    assert not isinstance(data._data["lines"], pd.DataFrame)

    # Act / Assert: Node data loaded
    pd.testing.assert_frame_equal(data["nodes"], nodes)
    assert isinstance(data._data["nodes"], pd.DataFrame)
    assert not isinstance(data._data["lines"], pd.DataFrame)

    # Act / Assert: Line data loaded
    pd.testing.assert_frame_equal(data["lines"], lines)
    assert isinstance(data._data["nodes"], pd.DataFrame)
    assert isinstance(data._data["lines"], pd.DataFrame)


def test_len(nodes: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes, lines=lines)

    # Act
    n_tables = len(data)

    # Assert
    assert n_tables == 2


def test_len__empty():
    # Arrange
    data = TabularData()

    # Act
    n_tables = len(data)

    # Assert
    assert n_tables == 0


def test_contains(nodes: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes, lines=lines)

    # Act / Assert
    assert "nodes" in data
    assert "lines" in data


def test_contains__empty():
    # Arrange
    data = TabularData()

    # Act / Assert
    assert "nodes" not in data


def test_keys(nodes: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes, lines=lines)

    # Act
    keys = list(data.keys())

    # Assert
    assert keys == ["nodes", "lines"]


def test_keys__empty():
    # Arrange
    data = TabularData()

    # Act
    keys = list(data.keys())

    # Assert
    assert keys == []


def test_items(nodes: pd.DataFrame, lines: pd.DataFrame):
    # Arrange
    data = TabularData(nodes=nodes, lines=lines)

    # Act
    items = list(data.items())

    # Assert
    assert items == [("nodes", nodes), ("lines", lines)]


def test_items__empty():
    # Arrange
    data = TabularData()

    # Act
    items = list(data.items())

    # Assert
    assert items == []
