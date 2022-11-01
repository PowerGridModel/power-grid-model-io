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
        self._create_pgm_input_source()
        # self._create_pgm_input_shunt()
        # self._create_pgm_input_transformer()
        # self._create_pgm_input_sym_gen()
        # self._create_pgm_input_three_winding_transformer()

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

        # Add an extra column called 'index' containing the index values, which we need to do the join with switches
        pp_lines["index"] = pp_lines.index

        # Select the appropriate switches and columns
        pp_switches = self.pp_data["switch"]
        pp_switches = pp_switches[self.pp_data["switch"]["et"] == "l"]
        pp_switches = pp_switches[["element", "bus", "closed"]]

        # Join the switches with the lines twice, once for the from_bus and once for the to_bus
        pp_from_switches = (
            pp_lines[["index", "from_bus"]]
            .merge(
                pp_switches,
                how="left",
                left_on=["index", "from_bus"],
                right_on=["element", "bus"],
            )
            .fillna(True)
            .set_index(pp_lines.index)
        )
        pp_to_switches = (
            pp_lines[["index", "to_bus"]]
            .merge(
                pp_switches,
                how="left",
                left_on=["index", "to_bus"],
                right_on=["element", "bus"],
            )
            .fillna(True)
            .set_index(pp_lines.index)
        )

        pgm_lines = initialize_array(data_type="input", component_type="line", shape=len(pp_lines))
        pgm_lines["id"] = self._generate_ids("line", pp_lines.index)
        pgm_lines["from_node"] = self._get_ids("bus", pp_lines["from_bus"])
        pgm_lines["from_status"] = pp_lines["in_service"] & pp_from_switches["closed"]
        pgm_lines["to_node"] = self._get_ids("bus", pp_lines["to_bus"])
        pgm_lines["to_status"] = pp_lines["in_service"] & pp_to_switches["closed"]
        pgm_lines["r1"] = pp_lines["r_ohm_per_km"] * pp_lines["length_km"]
        pgm_lines["x1"] = pp_lines["x_ohm_per_km"] * pp_lines["length_km"]
        pgm_lines["c1"] = pp_lines["c_nf_per_km"] * pp_lines["length_km"]
        pgm_lines["tan1"] = 0.0
        pgm_lines["i_n"] = pp_lines["max_i_ka"] * 1e3

        self.pgm_data["line"] = pgm_lines

    def _create_pgm_input_source(self):
        assert "source" not in self.pgm_data

        pp_ext_grid = self.pp_data["ext_grid"]

        pgm_source = initialize_array(data_type="input", component_type="source", shape=len(pp_ext_grid))
        pgm_source["id"] = self._generate_ids("source", pp_ext_grid.index)
        pgm_source["node"] = pp_ext_grid["bus"]
        pgm_source["status"] = pp_ext_grid["in_service"]
        pgm_source["u_ref"] = pp_ext_grid["vm_pu"]

    def _create_pgm_input_shunt(self):
        assert "shunt" not in self.pgm_data

        pp_shunt = self.pp_data["shunt"]

        pgm_shunt = initialize_array(data_type="input", component_type="shunt", shape=len(pp_shunt))
        pgm_shunt["id"] = self._generate_ids("shunt", pp_shunt.index)
        pgm_shunt["node"] = pp_shunt["bus"]
        pgm_shunt["status"] = pp_shunt["in_service"]
        pgm_shunt["g1"] = (pp_shunt["p_mw"] * 1e6) / ((pp_shunt["vn_kv"] * 1e3) * (pp_shunt["vn_kv"] * 1e3))
        pgm_shunt["b1"] = (pp_shunt["q_mvar"] * 1e6) / ((pp_shunt["vn_kv"] * 1e3) * (pp_shunt["vn_kv"] * 1e3))

    def _create_pgm_input_sym_gen(self):
        assert "sym_gen" not in self.pgm_data

        pp_sgen = self.pp_data["sgen"]

        pgm_sym_gen = initialize_array(data_type="input", component_type="sym_gen", shape=len(pp_sgen))
        pgm_sym_gen["id"] = self._generate_ids("sym_gen", pp_sgen.index)
        pgm_sym_gen["node"] = pp_sgen["bus"]
        pgm_sym_gen["status"] = pp_sgen["in_service"]
        pgm_sym_gen["p_specified"] = pp_sgen["p_mw"] * 1e6
        pgm_sym_gen["q_specified"] = pp_sgen["q_mvar"] * 1e6

    def _create_pgm_input_transformer(self):
        assert "transformer" not in self.pgm_data

        pp_trafo = self.pp_data["trafo"]

        pgm_transformer = initialize_array(data_type="input", component_type="transformer", shape=len(pp_trafo))
        pgm_transformer["id"] = self._generate_ids("transformer", pp_trafo.index)
        pgm_transformer["from_node"] = pp_trafo["hv_bus"]
        pgm_transformer["from_status"] = pp_trafo["in_service"]
        pgm_transformer["to_node"] = pp_trafo["lv_bus"]
        pgm_transformer["to_status"] = pp_trafo["in_service"]
        pgm_transformer["u1"] = pp_trafo["vn_hv_kv"] * 1e3
        pgm_transformer["u2"] = pp_trafo["vn_lv_kv"] * 1e3
        pgm_transformer["sn"] = pp_trafo["sn_mva"] * 1e6
        pgm_transformer["uk"] = pp_trafo["vkr_percent"] * 1e-2
        pgm_transformer["pk"] = (pp_trafo["vkr_percent"] * 1e-2) * (pp_trafo["sn_mva"] * 1e6)
        pgm_transformer["i0"] = pp_trafo["i0_percent"] * 1e-2
        pgm_transformer["p0"] = pp_trafo["pfe_kw"] * 1e3
        # pgm_transformer["winding_from"] =  vectorization?
        # pgm_transformer["winding_to"] =  vectorization?
        # pgm_transformer["clock"] =  vectorization?
        pgm_transformer["tap_pos"] = pp_trafo["tap_pos"]
        pgm_transformer["tap_side"] = pp_trafo["tap_side"]
        pgm_transformer["tap_min"] = pp_trafo["tap_min"]
        pgm_transformer["tap_max"] = pp_trafo["tap_max"]
        pgm_transformer["tap_nom"] = pp_trafo["tap_neutral"]
        # pgm_transformer["tap_size"] =  vectorization?

    def _create_pgm_input_three_winding_transformer(self):
        assert "three_winding_transformer" not in self.pgm_data

        pp_trafo3w = self.pp_data["trafo3w"]

        pgm_3wtransformer = initialize_array(
            data_type="input", component_type="three_winding_transformer", shape=len(pp_trafo3w)
        )
        pgm_3wtransformer["id"] = self._generate_ids("three_winding_transformer", pp_trafo3w.index)
        pgm_3wtransformer["node_1"] = pp_trafo3w["hv_bus"]
        pgm_3wtransformer["node_2"] = pp_trafo3w["mv_bus"]
        pgm_3wtransformer["node_3"] = pp_trafo3w["lv_bus"]
        pgm_3wtransformer["status_1"] = pp_trafo3w["in_service"]
        pgm_3wtransformer["status_2"] = pp_trafo3w["in_service"]
        pgm_3wtransformer["status_3"] = pp_trafo3w["in_service"]
        pgm_3wtransformer["u1"] = pp_trafo3w["vn_hv_kv"] * 1e3
        pgm_3wtransformer["u2"] = pp_trafo3w["vn_mv_kv"] * 1e3
        pgm_3wtransformer["u3"] = pp_trafo3w["vn_lv_kv"] * 1e3
        pgm_3wtransformer["sn_1"] = pp_trafo3w["sn_hv_mva"] * 1e6
        pgm_3wtransformer["sn_2"] = pp_trafo3w["sn_mv_mva"] * 1e6
        pgm_3wtransformer["sn_3"] = pp_trafo3w["sn_lv_mva"] * 1e6
        pgm_3wtransformer["uk_12"] = pp_trafo3w["vk_hv_percent"] * 1e-2
        pgm_3wtransformer["uk_13"] = pp_trafo3w["vk_lv_percent"] * 1e-2
        pgm_3wtransformer["uk_23"] = pp_trafo3w["vk_mv_percent"] * 1e-2

        pgm_3wtransformer["pk_12"] = (pp_trafo3w["vkr_hv_percent"] * 1e-2) * min(
            (pp_trafo3w["sn_hv_mva"] * 1e6), (pp_trafo3w["sn_mv_mva"] * 1e6)
        )

        pgm_3wtransformer["pk_13"] = (pp_trafo3w["vkr_lv_percent"] * 1e-2) * min(
            (pp_trafo3w["sn_hv_mva"] * 1e6), (pp_trafo3w["sn_lv_mva"] * 1e6)
        )

        pgm_3wtransformer["pk_23"] = (pp_trafo3w["vkr_mv_percent"] * 1e-2) * min(
            (pp_trafo3w["sn_mv_mva"] * 1e6), (pp_trafo3w["sn_lv_mva"] * 1e6)
        )

        pgm_3wtransformer["io"] = pp_trafo3w["i0_percent"] * 1e-2
        pgm_3wtransformer["p0"] = pp_trafo3w["pfe_kw"] * 1e3
        # pgm_transformer["winding_1"] =  vectorization?
        # pgm_transformer["winding_2"] =  vectorization?
        # pgm_transformer["winding_3"] =  vectorization?
        # pgm_transformer["clock_12"] =  vectorization?
        # pgm_transformer["clock_13"] =  vectorization?
        pgm_3wtransformer["tap_pos"] = pp_trafo3w["tap_pos"]
        pgm_3wtransformer["tap_side"] = pp_trafo3w["tap_side"]
        pgm_3wtransformer["tap_min"] = pp_trafo3w["tap_min"]
        pgm_3wtransformer["tap_max"] = pp_trafo3w["tap_max"]
        pgm_3wtransformer["tap_nom"] = pp_trafo3w["tap_neutral"]
        # pgm_transformer["tap_size"] =  vectorization?

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
