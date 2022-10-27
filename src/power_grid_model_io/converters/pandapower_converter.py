# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Panda Power Converter
"""
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from power_grid_model import initialize_array
from power_grid_model.data_types import Dataset

from power_grid_model_io.converters.base_converter import BaseConverter
from power_grid_model_io.data_types import ExtraInfoLookup

StdTypes = Dict[str, Dict[str, Dict[str, Union[float, int, str]]]]
PandasData = Dict[str, pd.DataFrame]


class PandaPowerConverter(BaseConverter[PandasData]):
    """
    Panda Power Converter
    """

    __slots__ = ("std_types", "pp_data", "pgm_data", "idx", "idx_lookup", "next_idx")

    def __init__(self, std_types: Optional[StdTypes] = None):
        super().__init__(source=None, destination=None)
        self.std_types: StdTypes = std_types if std_types is not None else {}
        self.pp_data: PandasData = {}
        self.pgm_data: Dataset = {}
        self.idx: Dict[str, pd.Series] = {}
        self.idx_lookup: Dict[str, pd.Series] = {}
        self.next_idx = 0

    def _parse_data(self, data: PandasData, data_type: str, extra_info: Optional[ExtraInfoLookup] = None) -> Dataset:

        # Clear pgm data
        self.pgm_data = {}
        self.idx_lookup = {}
        self.next_idx = 0

        # Set pandas data
        self.pp_data = data

        # Convert
        if data_type == "input":
            self._create_input_data()
        else:
            raise NotImplementedError()

        # Construct extra_info
        if extra_info is not None:
            pass
            # TODO: Construct extra info from self.idx_lookup
            # {
            #     0: {"table": "bus", "index": 1},
            #     1: {"table": "bus", "index": 2}
            #  }

        return self.pgm_data

    def _serialize_data(self, data: Dataset, extra_info: Optional[ExtraInfoLookup] = None) -> PandasData:
        raise NotImplementedError()

    def _create_input_data(self):
        self._create_pgm_input_nodes()
        self._create_pgm_input_lines()

    def _create_pgm_input_nodes(self):
        assert "node" not in self.pgm_data

        pp_busses = self.pp_data["bus"]
        pgm_nodes = initialize_array(data_type="input", component_type="node", shape=len(pp_busses))
        pgm_nodes["id"] = self._generate_ids("bus", pp_busses.index)
        pgm_nodes["u_rated"] = pp_busses["vn_kv"] * 1e3

        self.pgm_data["node"] = pgm_nodes

    def _create_pgm_input_lines(self):
        assert "line" not in self.pgm_data

        pp_lines = self.pp_data["line"]
        pp_switches = self.pp_data["switch"][self.pp_data["switch"]["et"] == "l"]

        pp_lines["index"] = pp_lines.index
        pp_from_switches = pp_lines[["index", "from_bus"]].merge(
            pp_switches[["element", "bus", "closed"]],
            how="left",
            left_on=["index", "from_bus"],
            right_on=["element", "bus"],
        )
        pp_to_switches = pp_lines[["index", "to_bus"]].merge(
            pp_switches[["element", "bus", "closed"]],
            how="left",
            left_on=["index", "to_bus"],
            right_on=["element", "bus"],
        )

        pgm_lines = initialize_array(data_type="input", component_type="line", shape=len(pp_lines))
        pgm_lines["id"] = self._generate_ids("line", pp_lines.index)
        pgm_lines["from_node"] = self._get_ids("bus", pp_lines["from_bus"])
        pgm_lines["from_status"] = pp_from_switches["closed"].fillna(1)
        pgm_lines["to_node"] = self._get_ids("bus", pp_lines["to_bus"])
        pgm_lines["to_status"] = pp_to_switches["closed"].fillna(1)
        pgm_lines["r1"] = pp_lines["r_ohm_per_km"] * pp_lines["length_km"]
        pgm_lines["x1"] = pp_lines["x_ohm_per_km"] * pp_lines["length_km"]
        pgm_lines["c1"] = pp_lines["c_nf_per_km"] * pp_lines["length_km"]
        pgm_lines["tan1"] = 0.0
        pgm_lines["i_n"] = pp_lines["max_i_ka"] * 1000.0

        self.pgm_data["line"] = pgm_lines

    def _generate_ids(self, key: str, pp_idx: pd.Index) -> np.arange:
        assert key not in self.idx_lookup
        n_objects = len(pp_idx)
        pgm_idx = np.arange(start=self.next_idx, stop=self.next_idx + n_objects, dtype=np.int32)
        self.idx[key] = pd.Series(pgm_idx, index=pp_idx)
        self.idx_lookup[key] = pd.Series(pp_idx, index=pgm_idx)
        self.next_idx += n_objects
        return pgm_idx

    def _get_ids(self, key: str, pp_idx: pd.Series) -> pd.Series:
        return self.idx[key][pp_idx]
