# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Panda Power Converter
"""
import math
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd
from power_grid_model import LoadGenType, initialize_array  # Branch3Side, BranchSide,
from power_grid_model.data_types import Dataset

from power_grid_model_io.converters.base_converter import BaseConverter
from power_grid_model_io.data_types import ExtraInfoLookup
from power_grid_model_io.filters.pandapower import (
    get_3wdgtransformer_winding_1,
    get_3wdgtransformer_winding_2,
    get_3wdgtransformer_winding_3,
    get_transformer_winding_from,
    get_transformer_winding_to,
)

StdTypes = Dict[str, Dict[str, Dict[str, Union[float, int, str]]]]
PandasData = Dict[str, pd.DataFrame]


# pylint: disable=too-many-instance-attributes
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
        self.pp_output_data: PandasData = {}
        self.pgm_output_data: Dataset = {}
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
        # Clear pp data
        self.pp_output_data = {}

        self.pgm_output_data = data

        # Convert
        self._create_output_data()

        # Construct extra_info
        if extra_info is not None:
            pass
            # TODO: Construct extra info from self.idx_lookup
            # {
            #     0: {"table": "bus", "index": 1},
            #     1: {"table": "bus", "index": 2}
            #  }

        return self.pp_output_data

    def _create_input_data(self):
        self._create_pgm_input_nodes()
        self._create_pgm_input_lines()
        self._create_pgm_input_sources()
        self._create_pgm_input_sym_loads()
        self._create_pgm_input_shunts()
        self._create_pgm_input_transformers()
        self._create_pgm_input_sym_gens()
        self._create_pgm_input_three_winding_transformers()

    def _create_output_data(self):
        # What about switches? loads?
        self._pp_buses_output()
        self._pp_lines_output()
        self._pp_ext_grids_output()
        self._pp_loads_output()  # Questionable
        self._pp_shunts_output()
        self._pp_trafos_output()
        self._pp_sgens_output()
        self._pp_trafos3w_output()

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
        pgm_shunts["node"] = self._get_ids("bus", pp_shunt["bus"])
        pgm_shunts["status"] = pp_shunt["in_service"]
        pgm_shunts["g1"] = (pp_shunt["p_mw"] * 1e6) / ((pp_shunt["vn_kv"] * 1e3) * (pp_shunt["vn_kv"] * 1e3))
        pgm_shunts["b1"] = (pp_shunt["q_mvar"] * 1e6) / ((pp_shunt["vn_kv"] * 1e3) * (pp_shunt["vn_kv"] * 1e3))

        self.pgm_data["shunt"] = pgm_shunts

    def _create_pgm_input_sym_gens(self):
        assert "sym_gen" not in self.pgm_data

        pp_sgens = self.pp_data["sgen"]

        pgm_sym_gens = initialize_array(data_type="input", component_type="sym_gen", shape=len(pp_sgens))
        pgm_sym_gens["id"] = self._generate_ids("sym_gen", pp_sgens.index)
        pgm_sym_gens["node"] = self._get_ids("bus", pp_sgens["bus"])
        pgm_sym_gens["status"] = pp_sgens["in_service"]
        pgm_sym_gens["p_specified"] = pp_sgens["p_mw"] * 1e6
        pgm_sym_gens["q_specified"] = pp_sgens["q_mvar"] * 1e6

        self.pgm_data["sym_gen"] = pgm_sym_gens

    def _create_pgm_input_sym_loads(self):
        assert "sym_load" not in self.pgm_data

        pp_loads = self.pp_data["load"]

        n_loads = len(pp_loads)

        pgm_sym_loads = initialize_array(data_type="input", component_type="sym_load", shape=3 * n_loads)

        const_i_multiplier = pp_loads["const_i_percent"] * (1e-2 * 1e6)
        const_z_multiplier = pp_loads["const_z_percent"] * (1e-2 * 1e6)
        const_p_multiplier = 1e6 - const_i_multiplier - const_z_multiplier

        pgm_sym_loads["id"][:n_loads] = self._generate_ids("sym_load_const_power", pp_loads.index)
        pgm_sym_loads["node"][:n_loads] = self._get_ids("bus", pp_loads["bus"])
        pgm_sym_loads["status"][:n_loads] = pp_loads["in_service"]
        pgm_sym_loads["type"][:n_loads] = LoadGenType.const_power
        pgm_sym_loads["p_specified"][:n_loads] = const_p_multiplier * pp_loads["p_mw"]
        pgm_sym_loads["q_specified"][:n_loads] = const_p_multiplier * pp_loads["q_mvar"]

        pgm_sym_loads["id"][n_loads : 2 * n_loads] = self._generate_ids("sym_load_const_impedance", pp_loads.index)
        pgm_sym_loads["node"][n_loads : 2 * n_loads] = self._get_ids("bus", pp_loads["bus"])
        pgm_sym_loads["status"][n_loads : 2 * n_loads] = pp_loads["in_service"]
        pgm_sym_loads["type"][n_loads : 2 * n_loads] = LoadGenType.const_impedance
        pgm_sym_loads["p_specified"][n_loads : 2 * n_loads] = const_z_multiplier * pp_loads["p_mw"]
        pgm_sym_loads["q_specified"][n_loads : 2 * n_loads] = const_z_multiplier * pp_loads["q_mvar"]

        pgm_sym_loads["id"][-n_loads:] = self._generate_ids("sym_load_const_current", pp_loads.index)
        pgm_sym_loads["node"][-n_loads:] = self._get_ids("bus", pp_loads["bus"])
        pgm_sym_loads["status"][-n_loads:] = pp_loads["in_service"]
        pgm_sym_loads["type"][-n_loads:] = LoadGenType.const_current
        pgm_sym_loads["p_specified"][-n_loads:] = const_i_multiplier * pp_loads["p_mw"]
        pgm_sym_loads["q_specified"][-n_loads:] = const_i_multiplier * pp_loads["q_mvar"]

        self.pgm_data["sym_load"] = pgm_sym_loads

    def _create_pgm_input_transformers(self):
        assert "transformer" not in self.pgm_data

        pp_trafo = self.pp_data["trafo"]

        pgm_transformers = initialize_array(data_type="input", component_type="transformer", shape=len(pp_trafo))
        pgm_transformers["id"] = self._generate_ids("transformer", pp_trafo.index)
        pgm_transformers["from_node"] = self._get_ids("bus", pp_trafo["hv_bus"])
        pgm_transformers["from_status"] = pp_trafo["in_service"]
        pgm_transformers["to_node"] = self._get_ids("bus", pp_trafo["lv_bus"])
        pgm_transformers["to_status"] = pp_trafo["in_service"]
        pgm_transformers["u1"] = pp_trafo["vn_hv_kv"] * 1e3
        pgm_transformers["u2"] = pp_trafo["vn_lv_kv"] * 1e3
        pgm_transformers["sn"] = pp_trafo["sn_mva"] * 1e6
        pgm_transformers["uk"] = pp_trafo["vkr_percent"] * 1e-2
        pgm_transformers["pk"] = (pp_trafo["vkr_percent"] * 1e-2) * (pp_trafo["sn_mva"] * 1e6)
        pgm_transformers["i0"] = pp_trafo["i0_percent"] * 1e-2
        pgm_transformers["p0"] = pp_trafo["pfe_kw"] * 1e3
        pgm_transformers["winding_from"] = pp_trafo["vector_group"].apply(get_transformer_winding_from)
        pgm_transformers["winding_to"] = pp_trafo["vector_group"].apply(get_transformer_winding_to)
        pgm_transformers["clock"] = ((pp_trafo["shift_degree"] / 30) % 12).astype(np.int8)
        pgm_transformers["tap_pos"] = pp_trafo["tap_pos"]
        pgm_transformers["tap_side"] = self._get_transformer_tap_side(pp_trafo["tap_side"])
        pgm_transformers["tap_min"] = pp_trafo["tap_min"]
        pgm_transformers["tap_max"] = pp_trafo["tap_max"]
        pgm_transformers["tap_nom"] = pp_trafo["tap_neutral"]
        pgm_transformers["tap_size"] = self._get_tap_size(pp_trafo)

        self.pgm_data["transformer"] = pgm_transformers

    def _create_pgm_input_three_winding_transformers(self):
        assert "three_winding_transformer" not in self.pgm_data

        pp_trafo3w = self.pp_data["trafo3w"]

        pgm_3wtransformers = initialize_array(
            data_type="input", component_type="three_winding_transformer", shape=len(pp_trafo3w)
        )
        pgm_3wtransformers["id"] = self._generate_ids("three_winding_transformer", pp_trafo3w.index)
        pgm_3wtransformers["node_1"] = self._get_ids("bus", pp_trafo3w["hv_bus"])
        pgm_3wtransformers["node_2"] = self._get_ids("bus", pp_trafo3w["mv_bus"])
        pgm_3wtransformers["node_3"] = self._get_ids("bus", pp_trafo3w["lv_bus"])
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

        pgm_3wtransformers["pk_12"] = (pp_trafo3w["vkr_hv_percent"] * 1e-2) * (
            pp_trafo3w[["sn_hv_mva", "sn_mv_mva"]].min(axis=1) * 1e6
        )

        pgm_3wtransformers["pk_13"] = (pp_trafo3w["vkr_lv_percent"] * 1e-2) * (
            pp_trafo3w[["sn_hv_mva", "sn_lv_mva"]].min(axis=1) * 1e6
        )

        pgm_3wtransformers["pk_23"] = (pp_trafo3w["vkr_mv_percent"] * 1e-2) * (
            pp_trafo3w[["sn_mv_mva", "sn_lv_mva"]].min(axis=1) * 1e6
        )

        pgm_3wtransformers["i0"] = pp_trafo3w["i0_percent"] * 1e-2
        pgm_3wtransformers["p0"] = pp_trafo3w["pfe_kw"] * 1e3
        pgm_3wtransformers["winding_1"] = pp_trafo3w["vector_group"].apply(get_3wdgtransformer_winding_1)
        pgm_3wtransformers["winding_2"] = pp_trafo3w["vector_group"].apply(get_3wdgtransformer_winding_2)
        pgm_3wtransformers["winding_3"] = pp_trafo3w["vector_group"].apply(get_3wdgtransformer_winding_3)
        pgm_3wtransformers["clock_12"] = (round(pp_trafo3w["shift_mv_degree"] / 30.0) % 12).astype(np.int8)
        pgm_3wtransformers["clock_13"] = (round(pp_trafo3w["shift_lv_degree"] / 30.0) % 12).astype(np.int8)
        pgm_3wtransformers["tap_pos"] = pp_trafo3w["tap_pos"]
        pgm_3wtransformers["tap_side"] = self._get_3wtransformer_tap_side(pp_trafo3w["tap_side"])
        pgm_3wtransformers["tap_min"] = pp_trafo3w["tap_min"]
        pgm_3wtransformers["tap_max"] = pp_trafo3w["tap_max"]
        pgm_3wtransformers["tap_nom"] = pp_trafo3w["tap_neutral"]
        pgm_3wtransformers["tap_size"] = self._get_3wtransformer_tap_size(pp_trafo3w)

        self.pgm_data["three_winding_transformer"] = pgm_3wtransformers

    def _pp_buses_output(self):  # different funciton name, we arent creating anything
        assert "bus" not in self.pp_output_data

        pgm_nodes = self.pgm_output_data["node"]  # we probably need result data and not input data

        pp_output_buses = pd.DataFrame(columns=["vm_pu", "va_degree", "p_mw", "q_mvar"])
        pp_output_buses["vm_pu"] = pgm_nodes["u_pu"]  # u_pu
        pp_output_buses["va_degree"] = self._get_degrees(
            pgm_nodes["u_angle"]
        )  # u_angle *180 /pi  def radians to degree
        # pp_output_buses["p_mw"] = # p_to and p_from connected to the bus have to be summed up
        # pp_output_buses["q_mvar"] = # q_to and q_from connected to the bus have to be summed up

        self.pp_output_data["bus"] = pp_output_buses

    def _pp_lines_output(self):
        assert "line" not in self.pp_output_data

        pgm_lines = self.pgm_output_data["line"]

        pp_output_lines = pd.DataFrame(
            columns=[
                "p_from_mw",
                "q_from_mvar",
                "p_to_mw",
                "q_to_mvar",
                "pl_mw",
                "ql_mvar",
                "i_from_ka",
                "i_to_ka",
                "i_ka",
                "vm_from_pu",
                "vm_to_pu",
                "va_from_degree",
                "va_to_degree",
                "loading_percent",
            ]
        )
        pp_output_lines["p_from_mw"] = pgm_lines["p_from"] * 1e-6  # p_from * 1e6
        pp_output_lines["q_from_mvar"] = pgm_lines["q_from"] * 1e-6  # q_from
        pp_output_lines["p_to_mw"] = pgm_lines["p_to"] * 1e-6  # p_to
        pp_output_lines["q_to_mvar"] = pgm_lines["q_to"] * 1e-6  # q_to
        pp_output_lines["pl_mw"] = (pgm_lines["p_from"] + pgm_lines["p_to"]) * 1e-6  # p_from + p_to
        pp_output_lines["ql_mvar"] = (pgm_lines["q_from"] + pgm_lines["q_to"]) * 1e-6  # q_from + q_to
        pp_output_lines["i_from_ka"] = pgm_lines["i_from"] * 1e-3  # i_from
        pp_output_lines["i_to_ka"] = pgm_lines["i_to"] * 1e-3  # i_to
        pp_output_lines["i_ka"] = pgm_lines[["i_from", "i_to"]].max(axis=1) * 1e-3  # max(i_from, i_to)
        # pp_output_lines["vm_from_pu"] = # u_pu of the bus that is at from_node
        # pp_output_lines["vm_to_pu"] = # u_pu of the bus that is at to_node
        # pp_output_lines["va_from_degree"] = # u_angle *180 /pi    def radians to degree/ at the from_bus
        # pp_output_lines["va_to_degree"] = # u_angle *180 /pi    def radians to degree/ at the to_bus
        pp_output_lines["loading_percent"] = pgm_lines["loading"] * 1e2  # loading * 100

        self.pp_output_data["line"] = pp_output_lines

    def _pp_ext_grids_output(self):
        assert "ext_grid" not in self.pp_output_data

        # pgm_sources = self.pgm_output_data["source"]

        pp_output_ext_grids = pd.DataFrame(columns=["p_mw", "q_mvar"])
        # pp_output_ext_grids["p_mw"] =
        # pp_output_ext_grids["q_mvar"] =

        self.pp_output_data["ext_grid"] = pp_output_ext_grids

    def _pp_shunts_output(self):
        assert "shunt" not in self.pp_output_data

        pgm_shunts = self.pgm_output_data["shunt"]

        pp_output_shunts = pd.DataFrame(columns=["p_mw", "q_mvar", "vm_pu"])
        pp_output_shunts["p_mw"] = pgm_shunts["p"] * 1e-6  # p
        pp_output_shunts["q_mvar"] = pgm_shunts["q"] * 1e-6  # q
        # pp_output_shunts["vm_pu"] = # u_pu at the bus at node

        self.pp_output_data["shunt"] = pp_output_shunts

    def _pp_sgens_output(self):
        assert "sgen" not in self.pp_output_data

        pgm_sym_gens = self.pgm_output_data["sym_gen"]

        pp_output_sgens = pd.DataFrame(columns=["p_mw", "q_mvar", "vm_pu"])
        pp_output_sgens["p_mw"] = pgm_sym_gens["p"] * 1e6  # p
        pp_output_sgens["q_mvar"] = pgm_sym_gens["q"] * 1e6  # q
        # pp_output_sgens["vm_pu"] = # u_pu at the bus at node

        self.pp_output_data["sgen"] = pp_output_sgens

    def _pp_trafos_output(self):
        assert "trafo" not in self.pp_output_data

        pgm_transformers = self.pgm_output_data["transformer"]

        pp_output_trafos = pd.DataFrame(
            columns=[
                "p_hv_mw",
                "q_hv_mvar",
                "p_lv_mw",
                "q_lv_mvar",
                "pl_mw",
                "ql_mvar",
                "i_hv_ka",
                "i_lv_ka",
                "vm_hv_pu",
                "vm_lv_pu",
                "va_hv_degree",
                "va_lv_degree",
                "loading_percent",
            ]
        )
        pp_output_trafos["p_hv_mw"] = pgm_transformers["p_from"] * 1e-6  # p_from
        pp_output_trafos["q_hv_mvar"] = pgm_transformers["q_from"] * 1e-6  # q_from
        pp_output_trafos["p_lv_mw"] = pgm_transformers["p_to"] * 1e-6  # p_to
        pp_output_trafos["q_lv_mvar"] = pgm_transformers["q_to"] * 1e-6  # q_to
        pp_output_trafos["pl_mw"] = (pgm_transformers["p_from"] + pgm_transformers["p_to"]) * 1e-6  # p_from + p_to
        pp_output_trafos["ql_mvar"] = (pgm_transformers["q_from"] + pgm_transformers["q_to"]) * 1e-6  # q_from + q_to
        pp_output_trafos["i_hv_ka"] = pgm_transformers["i_from"] * 1e-3  # i_from
        pp_output_trafos["i_lv_ka"] = pgm_transformers["i_to"] * 1e-3  # i_to
        # pp_output_trafos["vm_hv_pu"] = # voltage at the bus at the from_node
        # pp_output_trafos["vm_lv_pu"] = # voltage at the bus at the to_node
        # pp_output_trafos["va_hv_degree"] = # u_angle at bus at the from_node
        # pp_output_trafos["va_lv_degree"] = # u_angle at bus at the to_node
        pp_output_trafos["loading_percent"] = pgm_transformers["loading"] * 1e2  # loading * 100

        self.pp_output_data["trafo"] = pp_output_trafos

    def _pp_trafos3w_output(self):
        assert "trafo3w" not in self.pp_output_data

        pgm_transformers3w = self.pgm_output_data["three_winding_transformer"]

        pp_output_trafos3w = pd.DataFrame(
            columns=[
                "p_hv_mw",
                "q_hv_mvar",
                "p_mv_mw",
                "q_mv_mvar",
                "p_lv_mw",
                "q_lv_mvar",
                "pl_mw",
                "ql_mvar",
                "i_hv_ka",
                "i_mv_ka",
                "i_lv_ka",
                "vm_hv_pu",
                "vm_mv_pu",
                "vm_lv_pu",
                "va_hv_degree",
                "va_mv_degree",
                "va_lv_degree",
                "loading_percent",
            ],
            index=self._get_pp_ids("three_winding_transformer", pgm_transformers3w["id"]),
        )

        pp_output_trafos3w["p_hv_mw"] = pgm_transformers3w["p_1"] * 1e-6  # p_1
        pp_output_trafos3w["q_hv_mvar"] = pgm_transformers3w["q_1"] * 1e-6  # q_1
        pp_output_trafos3w["p_mv_mw"] = pgm_transformers3w["p_2"] * 1e-6  # p_2
        pp_output_trafos3w["q_mv_mvar"] = pgm_transformers3w["q_2"] * 1e-6  # q_2
        pp_output_trafos3w["p_lv_mw"] = pgm_transformers3w["p_3"] * 1e-6  # p_3
        pp_output_trafos3w["q_lv_mvar"] = pgm_transformers3w["q_e"] * 1e-6  # q_3
        pp_output_trafos3w["pl_mw"] = (
            pgm_transformers3w["p_1"] + pgm_transformers3w["p_2"] + pgm_transformers3w["p_3"]
        ) * 1e-6  # p_1 + p_2 + p_3
        pp_output_trafos3w["ql_mvar"] = (
            pgm_transformers3w["p_1"] + pgm_transformers3w["p_2"] + pgm_transformers3w["p_3"]
        ) * 1e-6  # q_1 + q_2 + q_3
        pp_output_trafos3w["i_hv_ka"] = pgm_transformers3w["i_1"] * 1e-3  # i_1
        pp_output_trafos3w["i_mv_ka"] = pgm_transformers3w["i_2"] * 1e-3  # i_2
        pp_output_trafos3w["i_lv_ka"] = pgm_transformers3w["i_3"] * 1e-3  # i_3
        # pp_output_trafos3w["vm_hv_pu"] = # voltage at the bus at the node_1
        # pp_output_trafos3w["vm_mv_pu"] = # voltage at the bus at the node_2
        # pp_output_trafos3w["vm_lv_pu"] = # voltage at the bus at the node_3
        # pp_output_trafos3w["va_hv_degree"] = # u_angle at bus at the node_1
        # pp_output_trafos3w["va_mv_degree"] = # u_angle at bus at the node_2
        # pp_output_trafos3w["va_lv_degree"] = # u_angle at bus at the node_3
        pp_output_trafos3w["loading_percent"] = pgm_transformers3w["loading"] * 1e2  # loading * 100

        self.pp_output_data["trafo3w"] = pp_output_trafos3w

    def _create_links(self):
        pass
        # switches which are bus to bus, they are links

    def _pp_loads_output(self):
        pass
        # total_loads = len()
        # pgm_data["sym_load"]
        # sym_load_p_id = self._get_pp_ids("sym_load_const_power", total_loads/3)
        # sym_load_p = sym_load[]
        # sym_load_z_id = self._get_pp_ids("sym_load_const_impedance", total_loads[])

    def _generate_ids(self, key: str, pp_idx: pd.Index) -> np.arange:
        assert key not in self.idx_lookup
        n_objects = len(pp_idx)
        pgm_idx = np.arange(start=self.next_idx, stop=self.next_idx + n_objects, dtype=np.int32)
        self.idx[key] = pd.Series(pgm_idx, index=pp_idx)
        self.idx_lookup[key] = pd.Series(pp_idx, index=pgm_idx)
        self.next_idx += n_objects
        return pgm_idx

    def _get_ids(self, key: str, pp_idx: pd.Series) -> pd.Series:
        if key not in self.idx:
            raise KeyError(f"No indexes have been created for '{key}'!")
        return self.idx[key][pp_idx]

    def _get_pp_ids(self, key: str, pgm_idx: pd.Series) -> pd.Series:
        if key not in self.idx_lookup:
            raise KeyError(f"No indexes have been created for '{key}'!")
        return self.idx_lookup[key][pgm_idx]

    @staticmethod
    def _get_tap_size(pp_trafo: pd.DataFrame) -> np.ndarray:
        tap_side_hv = np.array(pp_trafo["tap_side"] == "hv")
        tap_side_lv = np.array(pp_trafo["tap_side"] == "lv")
        tap_step_multiplier = pp_trafo["tap_step_percent"] * (1e-2 * 1e3)

        tap_size = np.empty(shape=len(pp_trafo), dtype=np.float64)
        tap_size[tap_side_hv] = tap_step_multiplier[tap_side_hv] * pp_trafo["vn_hv_kv"][tap_side_hv]
        tap_size[tap_side_lv] = tap_step_multiplier[tap_side_lv] * pp_trafo["vn_lv_kv"][tap_side_lv]

        return tap_size

    @staticmethod
    def _get_transformer_tap_side(tap_side: pd.Series) -> np.ndarray:
        new_tap_side = np.array(tap_side)
        new_tap_side[new_tap_side == "hv"] = 0
        new_tap_side[new_tap_side == "lv"] = 1

        return new_tap_side

    @staticmethod
    def _get_3wtransformer_tap_side(tap_side: pd.Series) -> np.ndarray:
        new_tap_side = np.array(tap_side)
        new_tap_side[new_tap_side == "hv"] = 0
        new_tap_side[new_tap_side == "mv"] = 1
        new_tap_side[new_tap_side == "lv"] = 2

        return new_tap_side

    @staticmethod
    def _get_3wtransformer_tap_size(pp_3wtrafo: pd.DataFrame) -> np.ndarray:
        tap_side_hv = np.array(pp_3wtrafo["tap_side"] == "hv")
        tap_side_mv = np.array(pp_3wtrafo["tap_side"] == "mv")
        tap_side_lv = np.array(pp_3wtrafo["tap_side"] == "lv")

        tap_step_multiplier = pp_3wtrafo["tap_step_percent"] * (1e-2 * 1e3)

        tap_size = np.empty(shape=len(pp_3wtrafo), dtype=np.float64)
        tap_size[tap_side_hv] = tap_step_multiplier[tap_side_hv] * pp_3wtrafo["vn_hv_kv"][tap_side_hv]
        tap_size[tap_side_mv] = tap_step_multiplier[tap_side_mv] * pp_3wtrafo["vn_mv_kv"][tap_side_mv]
        tap_size[tap_side_lv] = tap_step_multiplier[tap_side_lv] * pp_3wtrafo["vn_lv_kv"][tap_side_lv]

        return tap_size

    @staticmethod
    def _get_degrees(radians: float) -> int:
        return int(radians * 180 / math.pi)
