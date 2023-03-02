# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
CSV Directory Store
"""

from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from power_grid_model_io.data_stores.base_data_store import BaseDataStore
from power_grid_model_io.data_types import LazyDataFrame, TabularData


class CsvDirStore(BaseDataStore[TabularData]):
    """
    CSV Directory Store

    The first row of each .csv file is expected to contain the column names, unless specified differently by an
    extension of this class.
    """

    __slots__ = ("_dir_path", "_csv_kwargs", "_header_rows")

    def __init__(self, dir_path: Path, **csv_kwargs):
        super().__init__()
        self._dir_path = Path(dir_path)
        self._csv_kwargs: Dict[str, Any] = csv_kwargs
        self._header_rows: List[int] = [0]

    def load(self) -> TabularData:
        """
        Create a lazy loader for all CSV files in a directory and store them in a TabularData instance.
        """

        def lazy_csv_loader(csv_path: Path) -> LazyDataFrame:
            def csv_loader():
                return pd.read_csv(filepath_or_buffer=csv_path, header=self._header_rows, **self._csv_kwargs)

            return csv_loader

        data: Dict[str, LazyDataFrame] = {}
        for path in self._dir_path.glob("*.csv"):
            data[path.stem] = lazy_csv_loader(path)

        return TabularData(**data)

    def save(self, data: TabularData) -> None:
        """
        Store each table in data as a separate CSV file
        """
        for table_name, table_data in data.items():
            table_data.to_csv(path_or_buf=self._dir_path / f"{table_name}.csv", **self._csv_kwargs)
