# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from power_grid_model_io.data_stores.csv_dir_store import CsvDirStore
from power_grid_model_io.data_types import TabularData


@pytest.fixture()
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp).resolve()


def touch(file_path: Path):
    open(file_path, "wb").close()


@patch("power_grid_model_io.data_stores.csv_dir_store.pd.read_csv")
def test_load(mock_read_csv: MagicMock, temp_dir: Path):
    # Arrange
    foo_data = MagicMock()
    bar_data = MagicMock()
    touch(temp_dir / "foo.csv")
    touch(temp_dir / "bar.csv")
    mock_read_csv.side_effect = (foo_data, bar_data)
    csv_dir = CsvDirStore(temp_dir, bla=True)

    # Act
    csv_data = csv_dir.load()

    # Assert
    mock_read_csv.assert_not_called()  # The csv data is not yet loaded
    assert csv_data["foo"] == foo_data
    assert csv_data["bar"] == bar_data
    mock_read_csv.assert_any_call(filepath_or_buffer=temp_dir / "foo.csv", header=[0], bla=True)
    mock_read_csv.assert_any_call(filepath_or_buffer=temp_dir / "bar.csv", header=[0], bla=True)


@patch("power_grid_model_io.data_stores.csv_dir_store.pd.DataFrame.to_csv")
def test_save(mock_to_csv: MagicMock, temp_dir):
    # Arrange
    foo_data = pd.DataFrame()
    bar_data = np.array([])
    data = TabularData(foo=foo_data, bar=bar_data)
    csv_dir = CsvDirStore(temp_dir, bla=True)

    # Act
    csv_dir.save(data)

    # Assert
    mock_to_csv.assert_any_call(path_or_buf=temp_dir / "foo.csv", bla=True)
    mock_to_csv.assert_any_call(path_or_buf=temp_dir / "bar.csv", bla=True)
