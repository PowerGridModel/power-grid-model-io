# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
The json file store
"""

import json
from pathlib import Path
from typing import Optional

from power_grid_model_io.data_stores.base_data_store import BaseDataStore
from power_grid_model_io.data_types import StructuredData
from power_grid_model_io.utils.json import JsonEncoder, compact_json_dump


class JsonFileStore(BaseDataStore[StructuredData]):
    """
    The json file store expects each json file to be eiter a single dictionary, or a list of dictonaries.
    """

    __slots__ = ("_indent", "_compact", "_file_path")

    def __init__(self, file_path: Path):
        super().__init__()
        self._indent: Optional[int] = 2
        self._compact: bool = True
        self._file_path: Path = Path(file_path)

        # Check JSON file name
        if file_path.suffix.lower() != ".json":
            raise ValueError(f"JsonFile file should be a .json file, {file_path.suffix} provided.")

    def set_indent(self, indent: Optional[int]) -> None:
        """
        Change the number of spaces used for each indent level (affects output only)

        Args:
            indent: Number of spaces for each indent level. None = all json on a single line.
        """
        self._indent = indent

    def set_compact(self, compact: bool) -> None:
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

        Args:
            compact: Boolean defining if the output should be stored compact or not
        """
        self._compact = compact

    def load(self) -> StructuredData:
        """
        Loads a JSON file, validates the structure and returns it in a native python data format

        Returns: StructuredData
        """
        with self._file_path.open(mode="r", encoding="utf-8") as file_pointer:
            data = json.load(file_pointer)
        self._validate(data=data)
        return data

    def save(self, data: StructuredData) -> None:
        """
        Saves the native python data format as a JSON file

        Args:
            data: StructuredData
        """
        self._validate(data=data)
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        with self._file_path.open(mode="w", encoding="utf-8") as file_pointer:
            if self._compact and self._indent:
                max_level = 3
                if isinstance(data, list):
                    max_level += 1
                compact_json_dump(data, file_pointer, indent=self._indent, max_level=max_level)
            else:
                json.dump(data, file_pointer, indent=self._indent, cls=JsonEncoder)

    def _validate(self, data: StructuredData) -> None:
        # The data should be either a dictionary, or a (possibly empty) list of dictionaries
        if not isinstance(data, (dict, list)):
            raise TypeError(f"Invalid data type for {type(self).__name__}: {type(data).__name__}")

        if isinstance(data, list) and any(not isinstance(x, dict) for x in data):
            type_names = sorted({type(x).__name__ for x in data})
            if len(type_names) == 1:
                type_str = type_names.pop()
            else:
                type_str = " | ".join(type_names)
            raise TypeError(f"Invalid data type for {type(self).__name__}: List[{type_str}]")
