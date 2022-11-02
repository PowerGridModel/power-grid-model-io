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
        self._create_pgm_input_sources()
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

    def _create_pgm_input_sources(self):
        assert "source" not in self.pgm_data

        pp_ext_grid = self.pp_data["ext_grid"]

        pgm_sources = initialize_array(data_type="input", component_type="source", shape=len(pp_ext_grid))
        pgm_sources["id"] = self._generate_ids("source", pp_ext_grid.index)
        pgm_sources["node"] = self._get_ids("bus", pp_ext_grid["bus"])
        pgm_sources["status"] = pp_ext_grid["in_service"]
        pgm_sources["u_ref"] = pp_ext_grid["vm_pu"]

        self.pgm_data["source"] = pgm_sources

    def _create_pgm_input_shunts(self):
        assert "shunt" not in self.pgm_data

        pp_shunt = self.pp_data["shunt"]

        pgm_shunts = initialize_array(data_type="input", component_type="shunt", shape=len(pp_shunt))
        pgm_shunts["id"] = self._generate_ids("shunt", pp_shunt.index)
        pgm_shunts["node"] = pp_shunt["bus"]
        pgm_shunts["status"] = pp_shunt["in_service"]
        pgm_shunts["g1"] = (pp_shunt["p_mw"] * 1e6) / ((pp_shunt["vn_kv"] * 1e3) * (pp_shunt["vn_kv"] * 1e3))
        pgm_shunts["b1"] = (pp_shunt["q_mvar"] * 1e6) / ((pp_shunt["vn_kv"] * 1e3) * (pp_shunt["vn_kv"] * 1e3))

        self.pgm_data["shunt"] = pgm_shunts

    def _create_pgm_input_sym_gens(self):
        assert "sym_gen" not in self.pgm_data

        pp_sgens = self.pp_data["sgen"]

        pgm_sym_gens = initialize_array(data_type="input", component_type="sym_gen", shape=len(pp_sgens))
        pgm_sym_gens["id"] = self._generate_ids("sym_gen", pp_sgens.index)
        pgm_sym_gens["node"] = pp_sgens["bus"]
        pgm_sym_gens["status"] = pp_sgens["in_service"]
        pgm_sym_gens["p_specified"] = pp_sgens["p_mw"] * 1e6
        pgm_sym_gens["q_specified"] = pp_sgens["q_mvar"] * 1e6

        self.pgm_data["sym_gen"] = pgm_sym_gens

    def _create_pgm_input_sym_loads(self):
        assert "sym_load" not in self.pgm_data

        pp_loads = self.pp_data["load"]

        # if condition == True
        pgm_sym_loads = initialize_array(data_type="input", component_type="sym_load", shape=len(pp_loads))
        pgm_sym_loads["id"] = self._generate_ids("sym_gen", pp_loads.index)
        pgm_sym_loads["node"] = pp_loads["bus"], 0, 0
        pgm_sym_loads["status"] = pp_loads["in_service"], 0, 0
        pgm_sym_loads["type"] = 0, 1, 2
        pgm_sym_loads["p_specified"] = pp_loads["p_mw"] * 1e6, 0, 0
        pgm_sym_loads["q_specified"] = pp_loads["q_mvar"] * 1e6, 0, 0

    def _create_pgm_input_transformers(self):
        assert "transformer" not in self.pgm_data

        pp_trafo = self.pp_data["trafo"]

        pgm_transformers = initialize_array(data_type="input", component_type="transformer", shape=len(pp_trafo))
        pgm_transformers["id"] = self._generate_ids("transformer", pp_trafo.index)
        pgm_transformers["from_node"] = pp_trafo["hv_bus"]
        pgm_transformers["from_status"] = pp_trafo["in_service"]
        pgm_transformers["to_node"] = pp_trafo["lv_bus"]
        pgm_transformers["to_status"] = pp_trafo["in_service"]
        pgm_transformers["u1"] = pp_trafo["vn_hv_kv"] * 1e3
        pgm_transformers["u2"] = pp_trafo["vn_lv_kv"] * 1e3
        pgm_transformers["sn"] = pp_trafo["sn_mva"] * 1e6
        pgm_transformers["uk"] = pp_trafo["vkr_percent"] * 1e-2
        pgm_transformers["pk"] = (pp_trafo["vkr_percent"] * 1e-2) * (pp_trafo["sn_mva"] * 1e6)
        pgm_transformers["i0"] = pp_trafo["i0_percent"] * 1e-2
        pgm_transformers["p0"] = pp_trafo["pfe_kw"] * 1e3
        # pgm_transformers["winding_from"] =  vectorization?
        # pgm_transformers["winding_to"] =  vectorization?
        # pgm_transformers["clock"] =  vectorization?
        pgm_transformers["tap_pos"] = pp_trafo["tap_pos"]
        pgm_transformers["tap_side"] = pp_trafo["tap_side"]  # problem here
        pgm_transformers["tap_min"] = pp_trafo["tap_min"]
        pgm_transformers["tap_max"] = pp_trafo["tap_max"]
        pgm_transformers["tap_nom"] = pp_trafo["tap_neutral"]
        # pgm_transformers["tap_size"] =  vectorization?

        self.pgm_data["transformer"] = pgm_transformers

    def _create_pgm_input_three_winding_transformers(self):
        assert "three_winding_transformer" not in self.pgm_data

        pp_trafo3w = self.pp_data["trafo3w"]

        pgm_3wtransformers = initialize_array(
            data_type="input", component_type="three_winding_transformer", shape=len(pp_trafo3w)
        )
        pgm_3wtransformers["id"] = self._generate_ids("three_winding_transformer", pp_trafo3w.index)
        pgm_3wtransformers["node_1"] = pp_trafo3w["hv_bus"]
        pgm_3wtransformers["node_2"] = pp_trafo3w["mv_bus"]
        pgm_3wtransformers["node_3"] = pp_trafo3w["lv_bus"]
        pgm_3wtransformers["status_1"] = pp_trafo3w["in_service"]
        pgm_3wtransformers["status_2"] = pp_trafo3w["in_service"]
        pgm_3wtransformers["status_3"] = pp_trafo3w["in_service"]
        pgm_3wtransformers["u1"] = pp_trafo3w["vn_hv_kv"] * 1e3
        pgm_3wtransformers["u2"] = pp_trafo3w["vn_mv_kv"] * 1e3
        pgm_3wtransformers["u3"] = pp_trafo3w["vn_lv_kv"] * 1e3
        pgm_3wtransformers["sn_1"] = pp_trafo3w["sn_hv_mva"] * 1e6
        pgm_3wtransformers["sn_2"] = pp_trafo3w["sn_mv_mva"] * 1e6
        pgm_3wtransformers["sn_3"] = pp_trafo3w["sn_lv_mva"] * 1e6
        pgm_3wtransformers["uk_12"] = pp_trafo3w["vk_hv_percent"] * 1e-2
        pgm_3wtransformers["uk_13"] = pp_trafo3w["vk_lv_percent"] * 1e-2
        pgm_3wtransformers["uk_23"] = pp_trafo3w["vk_mv_percent"] * 1e-2

        pgm_3wtransformers["pk_12"] = (pp_trafo3w["vkr_hv_percent"] * 1e-2) * min(
            (pp_trafo3w["sn_hv_mva"] * 1e6), (pp_trafo3w["sn_mv_mva"] * 1e6)
        )

        pgm_3wtransformers["pk_13"] = (pp_trafo3w["vkr_lv_percent"] * 1e-2) * min(
            (pp_trafo3w["sn_hv_mva"] * 1e6), (pp_trafo3w["sn_lv_mva"] * 1e6)
        )

        pgm_3wtransformers["pk_23"] = (pp_trafo3w["vkr_mv_percent"] * 1e-2) * min(
            (pp_trafo3w["sn_mv_mva"] * 1e6), (pp_trafo3w["sn_lv_mva"] * 1e6)
        )

        pgm_3wtransformers["io"] = pp_trafo3w["i0_percent"] * 1e-2
        pgm_3wtransformers["p0"] = pp_trafo3w["pfe_kw"] * 1e3
        # pgm_transformers["winding_1"] =  vectorization?
        # pgm_transformers["winding_2"] =  vectorization?
        # pgm_transformers["winding_3"] =  vectorization?
        # pgm_transformers["clock_12"] =  vectorization?
        # pgm_transformers["clock_13"] =  vectorization?
        pgm_3wtransformers["tap_pos"] = pp_trafo3w["tap_pos"]
        pgm_3wtransformers["tap_side"] = pp_trafo3w["tap_side"]  # problem here
        pgm_3wtransformers["tap_min"] = pp_trafo3w["tap_min"]
        pgm_3wtransformers["tap_max"] = pp_trafo3w["tap_max"]
        pgm_3wtransformers["tap_nom"] = pp_trafo3w["tap_neutral"]
        # pgm_transformers["tap_size"] =  vectorization?

        self.pgm_data["three_winding_transformer"] = pgm_3wtransformers

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
