# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
The rawx json file store
"""

import json
from pathlib import Path

import pandas as pd

from power_grid_model_io.data_stores.base_data_store import BaseDataStore
from power_grid_model_io.data_types import RawxData

COMPONENT_DATA = dict[str, list | list[list]]
NETWORK_DATA = dict[str, COMPONENT_DATA]
RAWX_DATA = dict[str, NETWORK_DATA]


class RawxFileStore(BaseDataStore[RawxData]):
    """

    Args:
        BaseDataStore (_type_): _description_
    """

    def __init__(self, file_path: Path):
        super().__init__()
        self._file_path = file_path

        if self._file_path.suffix.lower() != ".rawx":
            raise ValueError(f"Input file should be a .rawx, {self._file_path.suffix} provided.")

    def load(self) -> RawxData:
        """
        Returns:
            RawxData: _description_
        """
        with self._file_path.open(mode="r", encoding="utf-8") as file_pointer:
            json_data = json.load(file_pointer)

        data = self._parse_rawx_data(json_data)
        return RawxData(**data)

    def _parse_rawx_data(self, json_data: RAWX_DATA) -> dict[str, pd.DataFrame]:
        data: dict[str, pd.DataFrame] = {}

        network = json_data["network"]

        for component in network:
            columns = network[component]["fields"]
            component_data = network[component]["data"]
            if len(component_data) == 0:
                data[component] = pd.DataFrame(columns=columns)
                continue
            if not isinstance(component_data[0], list):
                component_data = [component_data]
            data[component] = pd.DataFrame(data=component_data, columns=columns)
        return data

    def save(self, _: RawxData):
        raise NotImplementedError("Save method not implemneted for RawxFileStore")
