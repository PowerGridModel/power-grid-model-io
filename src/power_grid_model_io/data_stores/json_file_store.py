# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
The json file store
"""

import json
from pathlib import Path
from typing import Optional

from power_grid_model.utils import compact_json_dump

from power_grid_model_io.data_stores.base_data_store import BaseDataStore
from power_grid_model_io.data_types import StructuredData


class JsonFileStore(BaseDataStore[StructuredData]):
    """
    The json file store expects each json file to be eiter a single dictionary, or a list of dictonaries.
    """

    __slots__ = ("_indent", "_compact", "_file_path")

    def __init__(self, file_path: Path):
        super().__init__()
        self._indent: Optional[int] = 2
        self._compact: bool = True
        self._file_path: Path = file_path

        # Check JSON file name
        if file_path.suffix.lower() != ".json":
            raise ValueError(f"JsonFile file should be a .json file, {file_path.suffix} provided.")

    def set_indent(self, indent: Optional[int]):
        """
        Change the number of spaced used for each indent level (affects output only)
        """
        self._indent = indent

    def set_compact(self, compact: bool):
        """
        In compact mode, each object will be output on a single line. Note that the JsonFileStore is not very
        general; it assumes that data is either a dictionary of this format:

        {
          "category_0":
            [
              {"attribute_0": ..., "attribute_1": ..., ...},
              {"attribute_0": ..., "attribute_1": ..., ...},
            ],
          "category_1":
            [
              {"attribute_0": ..., "attribute_1": ..., ...},
              {"attribute_0": ..., "attribute_1": ..., ...},
            ],
          ...
        }

        or a list of those dictionaries.
        """
        self._compact = compact

    def load(self) -> StructuredData:
        with self._file_path.open(mode="r", encoding="utf-8") as file_pointer:
            data = json.load(file_pointer)
        self._validate(data=data)
        return data

    def save(self, data: StructuredData) -> None:
        self._validate(data=data)
        if self._compact and self._indent:
            max_level = 3
            if isinstance(data, list):
                max_level += 1
            with self._file_path.open(mode="w", encoding="utf-8") as file_pointer:
                compact_json_dump(data, file_pointer, indent=self._indent, max_level=max_level)
        else:
            with self._file_path.open(mode="w", encoding="utf-8") as file_pointer:
                json.dump(data, file_pointer, indent=self._indent)
