# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Panda Power Converter
"""
import re
from functools import lru_cache
from typing import Dict, Mapping, Optional, Union

import numpy as np
import pandas as pd
from power_grid_model import Branch3Side, BranchSide, LoadGenType, initialize_array
from power_grid_model.data_types import Dataset

from power_grid_model_io.converters.base_converter import BaseConverter
from power_grid_model_io.data_types import ExtraInfoLookup
from power_grid_model_io.functions import get_winding

StdTypes = Mapping[str, Mapping[str, Mapping[str, Union[float, int, str]]]]
PandaPowerData = Mapping[str, pd.DataFrame]

CONNECTION_PATTERN_PP = re.compile(r"(Y|YN|D|Z|ZN)(y|yn|d|z|zn)\d*")
CONNECTION_PATTERN_PP_3WDG = re.compile(r"(Y|YN|D|Z|ZN)(y|yn|d|z|zn)(y|yn|d|z|zn)\d*")


class PandaPowerConverter(BaseConverter[PandaPowerData]):
    """
    Panda Power Converter
    """

    __slots__ = ("std_types", "pp_data", "pgm_data", "idx", "idx_lookup", "next_idx", "system_frequency")

    def __init__(self, std_types: Optional[StdTypes] = None, system_frequency: float = 50.0):
        super().__init__(source=None, destination=None)
        self.std_types: StdTypes = std_types if std_types is not None else {}
        self.system_frequency: float = system_frequency
        self.pp_data: PandaPowerData = {}
        self.pgm_data: Dataset = {}
        self.idx: Dict[str, pd.Series] = {}
        self.idx_lookup: Dict[str, pd.Series] = {}
        self.next_idx = 0

    def _parse_data(
        self, data: PandaPowerData, data_type: str, extra_info: Optional[ExtraInfoLookup] = None
    ) -> Dataset:

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
            raise ValueError(f"Data type: '{data_type}' is not implemented")

        # Construct extra_info
        if extra_info is not None:
            for pp_table, indices in self.idx_lookup.items():
                for pgm_idx, pp_idx in zip(indices.index, indices):
                    extra_info[pgm_idx] = {"id_reference": {"table": pp_table, "index": pp_idx}}

        return self.pgm_data

    def _serialize_data(self, data: Dataset, extra_info: Optional[ExtraInfoLookup]) -> PandaPowerData:
        raise NotImplementedError()

    def _create_input_data(self):
        self._create_pgm_input_nodes()
        self._create_pgm_input_lines()
        self._create_pgm_input_sources()
        self._create_pgm_input_sym_loads()
        self._create_pgm_input_shunts()
        self._create_pgm_input_transformers()
        self._create_pgm_input_sym_gens()
        self._create_pgm_input_three_winding_transformers()
        self._create_pgm_input_links()

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

        switch_states = self.get_switch_states("line")

        pgm_lines = initialize_array(data_type="input", component_type="line", shape=len(pp_lines))
        pgm_lines["id"] = self._generate_ids("line", pp_lines.index)
        pgm_lines["from_node"] = self._get_ids("bus", pp_lines["from_bus"])
        pgm_lines["from_status"] = pp_lines["in_service"] & switch_states.iloc[0, :]
        pgm_lines["to_node"] = self._get_ids("bus", pp_lines["to_bus"])
        pgm_lines["to_status"] = pp_lines["in_service"] & switch_states.iloc[1, :]
        pgm_lines["r1"] = pp_lines["r_ohm_per_km"] * pp_lines["length_km"] / pp_lines["parallel"]
        pgm_lines["x1"] = pp_lines["x_ohm_per_km"] * pp_lines["length_km"] / pp_lines["parallel"]
        pgm_lines["c1"] = pp_lines["c_nf_per_km"] * pp_lines["length_km"] * pp_lines["parallel"] * 1e-9
        # The formula for tan1 = R_1 / Xc_1 = (g * 1e-6) / (2 * pi * f * c * 1e-9) = g / (2 * pi * f * c * 1e-3)
        pgm_lines["tan1"] = (
            pp_lines["g_us_per_km"] / pp_lines["c_nf_per_km"] / (2 * np.pi * self.system_frequency * 1e-3)
        )
        pgm_lines["i_n"] = (pp_lines["max_i_ka"] * 1e3) * pp_lines["df"] * pp_lines["parallel"]

        self.pgm_data["line"] = pgm_lines

    def _create_pgm_input_sources(self):
        assert "source" not in self.pgm_data

        pp_ext_grid = self.pp_data["ext_grid"]

        pgm_sources = initialize_array(data_type="input", component_type="source", shape=len(pp_ext_grid))
        pgm_sources["id"] = self._generate_ids("ext_grid", pp_ext_grid.index)
        pgm_sources["node"] = self._get_ids("bus", pp_ext_grid["bus"])
        pgm_sources["status"] = pp_ext_grid["in_service"]
        pgm_sources["u_ref"] = pp_ext_grid["vm_pu"]
        pgm_sources["rx_ratio"] = pp_ext_grid["rx_max"]
        pgm_sources["u_ref_angle"] = pp_ext_grid["va_degree"] * (np.pi / 180)

        if "s_sc_max_mva" in pp_ext_grid:
            pgm_sources["sk"] = pp_ext_grid["s_sc_max_mva"] * 1e6

        self.pgm_data["source"] = pgm_sources

    def _create_pgm_input_shunts(self):
        assert "shunt" not in self.pgm_data

        pp_shunt = self.pp_data["shunt"]

        vn_kv_2 = pp_shunt["vn_kv"] * pp_shunt["vn_kv"]

        pgm_shunts = initialize_array(data_type="input", component_type="shunt", shape=len(pp_shunt))
        pgm_shunts["id"] = self._generate_ids("shunt", pp_shunt.index)
        pgm_shunts["node"] = self._get_ids("bus", pp_shunt["bus"])
        pgm_shunts["status"] = pp_shunt["in_service"]
        pgm_shunts["g1"] = pp_shunt["p_mw"] * pp_shunt["step"] / vn_kv_2
        pgm_shunts["b1"] = -pp_shunt["q_mvar"] * pp_shunt["step"] / vn_kv_2

        self.pgm_data["shunt"] = pgm_shunts

    def _create_pgm_input_sym_gens(self):
        assert "sym_gen" not in self.pgm_data

        pp_sgens = self.pp_data["sgen"]

        pgm_sym_gens = initialize_array(data_type="input", component_type="sym_gen", shape=len(pp_sgens))
        pgm_sym_gens["id"] = self._generate_ids("sgen", pp_sgens.index)
        pgm_sym_gens["node"] = self._get_ids("bus", pp_sgens["bus"])
        pgm_sym_gens["status"] = pp_sgens["in_service"]
        pgm_sym_gens["p_specified"] = pp_sgens["p_mw"] * 1e6 * pp_sgens["scaling"]
        pgm_sym_gens["q_specified"] = pp_sgens["q_mvar"] * 1e6 * pp_sgens["scaling"]
        pgm_sym_gens["type"] = LoadGenType.const_power

        self.pgm_data["sym_gen"] = pgm_sym_gens

    def _create_pgm_input_sym_loads(self):
        assert "sym_load" not in self.pgm_data

        pp_loads = self.pp_data["load"]

        n_loads = len(pp_loads)

        pgm_sym_loads = initialize_array(data_type="input", component_type="sym_load", shape=3 * n_loads)

        const_i_multiplier = pp_loads["const_i_percent"] * pp_loads["scaling"] * (1e-2 * 1e6)
        const_z_multiplier = pp_loads["const_z_percent"] * pp_loads["scaling"] * (1e-2 * 1e6)
        const_p_multiplier = (1e6 - const_i_multiplier - const_z_multiplier) * pp_loads["scaling"]

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

        switch_states = self.get_switch_states("trafo")
        winding_types = self.get_trafo_winding_types()

        pgm_transformers = initialize_array(data_type="input", component_type="transformer", shape=len(pp_trafo))
        pgm_transformers["id"] = self._generate_ids("trafo", pp_trafo.index)
        pgm_transformers["from_node"] = self._get_ids("bus", pp_trafo["hv_bus"])
        pgm_transformers["from_status"] = pp_trafo["in_service"] & switch_states.iloc[0, :]
        pgm_transformers["to_node"] = self._get_ids("bus", pp_trafo["lv_bus"])
        pgm_transformers["to_status"] = pp_trafo["in_service"] & switch_states.iloc[1, :]
        pgm_transformers["u1"] = pp_trafo["vn_hv_kv"] * 1e3
        pgm_transformers["u2"] = pp_trafo["vn_lv_kv"] * 1e3
        pgm_transformers["sn"] = pp_trafo["sn_mva"] * pp_trafo["parallel"] * 1e6
        pgm_transformers["uk"] = pp_trafo["vk_percent"] * 1e-2
        pgm_transformers["pk"] = pp_trafo["vkr_percent"] * pp_trafo["sn_mva"] * pp_trafo["parallel"] * (1e6 * 1e-2)
        pgm_transformers["i0"] = pp_trafo["i0_percent"] * 1e-2
        pgm_transformers["p0"] = pp_trafo["pfe_kw"] * pp_trafo["parallel"] * 1e3
        pgm_transformers["winding_from"] = winding_types["winding_from"]
        pgm_transformers["winding_to"] = winding_types["winding_to"]
        pgm_transformers["clock"] = round(pp_trafo["shift_degree"] / 30) % 12
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

        switch_states = self.get_trafo3w_switch_states(pp_trafo3w)
        winding_type = self.get_trafo3w_winding_types()

        pgm_3wtransformers = initialize_array(
            data_type="input", component_type="three_winding_transformer", shape=len(pp_trafo3w)
        )
        pgm_3wtransformers["id"] = self._generate_ids("trafo3w", pp_trafo3w.index)
        pgm_3wtransformers["node_1"] = self._get_ids("bus", pp_trafo3w["hv_bus"])
        pgm_3wtransformers["node_2"] = self._get_ids("bus", pp_trafo3w["mv_bus"])
        pgm_3wtransformers["node_3"] = self._get_ids("bus", pp_trafo3w["lv_bus"])
        pgm_3wtransformers["status_1"] = pp_trafo3w["in_service"] & switch_states.iloc[0, :]
        pgm_3wtransformers["status_2"] = pp_trafo3w["in_service"] & switch_states.iloc[1, :]
        pgm_3wtransformers["status_3"] = pp_trafo3w["in_service"] & switch_states.iloc[2, :]
        pgm_3wtransformers["u1"] = pp_trafo3w["vn_hv_kv"] * 1e3
        pgm_3wtransformers["u2"] = pp_trafo3w["vn_mv_kv"] * 1e3
        pgm_3wtransformers["u3"] = pp_trafo3w["vn_lv_kv"] * 1e3
        pgm_3wtransformers["sn_1"] = pp_trafo3w["sn_hv_mva"] * 1e6
        pgm_3wtransformers["sn_2"] = pp_trafo3w["sn_mv_mva"] * 1e6
        pgm_3wtransformers["sn_3"] = pp_trafo3w["sn_lv_mva"] * 1e6
        pgm_3wtransformers["uk_12"] = pp_trafo3w["vk_hv_percent"] * 1e-2
        pgm_3wtransformers["uk_13"] = pp_trafo3w["vk_lv_percent"] * 1e-2
        pgm_3wtransformers["uk_23"] = pp_trafo3w["vk_mv_percent"] * 1e-2

        pgm_3wtransformers["pk_12"] = (
            pp_trafo3w["vkr_hv_percent"] * (pp_trafo3w[["sn_hv_mva", "sn_mv_mva"]].min(axis=1)) * (1e-2 * 1e6)
        )

        pgm_3wtransformers["pk_13"] = (
            pp_trafo3w["vkr_lv_percent"] * (pp_trafo3w[["sn_hv_mva", "sn_lv_mva"]].min(axis=1)) * (1e-2 * 1e6)
        )

        pgm_3wtransformers["pk_23"] = (
            pp_trafo3w["vkr_mv_percent"] * (pp_trafo3w[["sn_mv_mva", "sn_lv_mva"]].min(axis=1)) * (1e-2 * 1e6)
        )

        pgm_3wtransformers["i0"] = pp_trafo3w["i0_percent"] * 1e-2
        pgm_3wtransformers["p0"] = pp_trafo3w["pfe_kw"] * 1e3
        pgm_3wtransformers["winding_1"] = winding_type["winding_1"]
        pgm_3wtransformers["winding_2"] = winding_type["winding_2"]
        pgm_3wtransformers["winding_3"] = winding_type["winding_3"]
        pgm_3wtransformers["clock_12"] = round(pp_trafo3w["shift_mv_degree"] / 30.0) % 12
        pgm_3wtransformers["clock_13"] = round(pp_trafo3w["shift_lv_degree"] / 30.0) % 12
        pgm_3wtransformers["tap_pos"] = pp_trafo3w["tap_pos"]
        pgm_3wtransformers["tap_side"] = self._get_3wtransformer_tap_side(pp_trafo3w["tap_side"])
        pgm_3wtransformers["tap_min"] = pp_trafo3w["tap_min"]
        pgm_3wtransformers["tap_max"] = pp_trafo3w["tap_max"]
        pgm_3wtransformers["tap_nom"] = pp_trafo3w["tap_neutral"]
        pgm_3wtransformers["tap_size"] = self._get_3wtransformer_tap_size(pp_trafo3w)

        self.pgm_data["three_winding_transformer"] = pgm_3wtransformers

    def _create_pgm_input_links(self):
        assert "link" not in self.pgm_data

        pp_switches = self.pp_data["switch"]
        pp_switches = pp_switches[
            self.pp_data["switch"]["et"] == "b"
        ]  # This should take all the switches which are b2b

        pgm_links = initialize_array(data_type="input", component_type="link", shape=len(pp_switches))
        pgm_links["id"] = self._generate_ids("b2b-switch", pp_switches.index)
        pgm_links["from_node"] = self._get_ids("bus", pp_switches["bus"])
        pgm_links["to_node"] = self._get_ids("bus", pp_switches["element"])
        pgm_links["from_status"] = pp_switches["closed"]
        pgm_links["to_status"] = pp_switches["closed"]

        self.pgm_data["link"] = pgm_links

    def _generate_ids(self, pp_table: str, pp_idx: pd.Index) -> np.arange:
        assert pp_table not in self.idx_lookup
        n_objects = len(pp_idx)
        pgm_idx = np.arange(start=self.next_idx, stop=self.next_idx + n_objects, dtype=np.int32)
        self.idx[pp_table] = pd.Series(pgm_idx, index=pp_idx)
        self.idx_lookup[pp_table] = pd.Series(pp_idx, index=pgm_idx)
        self.next_idx += n_objects
        return pgm_idx

    def _get_ids(self, pp_table: str, pp_idx: pd.Series) -> pd.Series:
        if pp_table not in self.idx:
            raise KeyError(f"No indexes have been created for '{pp_table}'!")
        return self.idx[pp_table][pp_idx]

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
        new_tap_side[new_tap_side == "hv"] = BranchSide.from_side
        new_tap_side[new_tap_side == "lv"] = BranchSide.to_side

        return new_tap_side

    @staticmethod
    def _get_3wtransformer_tap_side(tap_side: pd.Series) -> np.ndarray:
        new_tap_side = np.array(tap_side)
        new_tap_side[new_tap_side == "hv"] = Branch3Side.side_1
        new_tap_side[new_tap_side == "mv"] = Branch3Side.side_2
        new_tap_side[new_tap_side == "lv"] = Branch3Side.side_3

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
    def get_individual_switch_states(component: pd.DataFrame, switches: pd.DataFrame, bus: str) -> pd.Series:
        """
        Return the state of individual switch. Can be open or closed.
        """
        switch_state = (
            component[["index", bus]]
            .merge(
                switches,
                how="left",
                left_on=["index", bus],
                right_on=["element", "bus"],
            )
            .fillna(True)
            .set_index(component.index)
        )

        return switch_state["closed"]

    def get_switch_states(self, pp_table: str) -> pd.DataFrame:
        """
        Return switch states of either lines or transformers
        """
        if pp_table == "line":
            element_type = "l"
            bus1 = "from_bus"
            bus2 = "to_bus"
        else:
            element_type = "t"
            bus1 = "hv_bus"
            bus2 = "lv_bus"

        component = self.pp_data[pp_table]
        component["index"] = component.index
        # Select the appropriate switches and columns
        pp_switches = self.pp_data["switch"]
        pp_switches = pp_switches[self.pp_data["switch"]["et"] == element_type]
        pp_switches = pp_switches[["element", "bus", "closed"]]

        pp_from_switches = self.get_individual_switch_states(component, pp_switches, bus1)
        pp_to_switches = self.get_individual_switch_states(component, pp_switches, bus2)

        return pd.DataFrame(data=(pp_from_switches, pp_to_switches))

    def get_trafo3w_switch_states(self, component: pd.DataFrame) -> pd.DataFrame:
        """
        Return switch states of three winding transformer
        """
        element_type = "t3"
        bus1 = "hv_bus"
        bus2 = "mv_bus"
        bus3 = "lv_bus"
        component["index"] = component.index

        # Select the appropriate switches and columns
        pp_switches = self.pp_data["switch"]
        pp_switches = pp_switches[self.pp_data["switch"]["et"] == element_type]
        pp_switches = pp_switches[["element", "bus", "closed"]]

        # Join the switches with the three winding trafo three times, for the hv_bus, mv_bus and once for the lv_bus
        pp_1_switches = self.get_individual_switch_states(component, pp_switches, bus1)
        pp_2_switches = self.get_individual_switch_states(component, pp_switches, bus2)
        pp_3_switches = self.get_individual_switch_states(component, pp_switches, bus3)

        return pd.DataFrame((pp_1_switches, pp_2_switches, pp_3_switches))

    def get_trafo_winding_types(self) -> pd.DataFrame:
        """
        Return the from and to winding type
        """

        @lru_cache
        def vector_group_to_winding_types(vector_group: str) -> pd.Series:
            match = CONNECTION_PATTERN_PP.fullmatch(vector_group)
            if not match:
                raise ValueError(f"Invalid transformer connection string: '{vector_group}'")
            winding_from = get_winding(match.group(1)).value
            winding_to = get_winding(match.group(2)).value
            return pd.Series([winding_from, winding_to])

        @lru_cache
        def std_type_to_winding_types(std_type: str) -> pd.Series:
            return vector_group_to_winding_types(self.std_types["trafo"][std_type]["vector_group"])

        trafo = self.pp_data["trafo"]
        if "vector_group" in trafo:
            trafo = trafo["vector_group"].apply(vector_group_to_winding_types)
        else:
            trafo = trafo["std_type"].apply(std_type_to_winding_types)
        trafo.columns = ["winding_from", "winding_to"]
        return trafo

    def get_trafo3w_winding_types(self) -> pd.DataFrame:
        """
        Return the three winding types
        """

        @lru_cache
        def vector_group_to_winding_types(vector_group: str) -> pd.Series:
            match = CONNECTION_PATTERN_PP_3WDG.fullmatch(vector_group)
            if not match:
                raise ValueError(f"Invalid transformer connection string: '{vector_group}'")
            winding_1 = get_winding(match.group(1)).value
            winding_2 = get_winding(match.group(2)).value
            winding_3 = get_winding(match.group(3)).value
            return pd.Series([winding_1, winding_2, winding_3])

        @lru_cache
        def std_type_to_winding_types(std_type: str) -> pd.Series:
            return vector_group_to_winding_types(self.std_types["trafo3w"][std_type]["vector_group"])

        trafo3w = self.pp_data["trafo3w"]
        if "vector_group" in trafo3w:
            trafo3w = trafo3w["vector_group"].apply(vector_group_to_winding_types)
        else:
            trafo3w = trafo3w["std_type"].apply(std_type_to_winding_types)
        trafo3w.columns = ["winding_1", "winding_2", "winding_3"]
        return trafo3w

    def get_id(self, pp_table: str, pp_idx: int) -> int:
        """
        Get a the numerical ID previously associated with the supplied table / index combination

        Args:
            pp_table: Table name (e.g. "bus")
            pp_idx: PandaPower component identifier

        Returns: The associated id
        """
        return self.idx[pp_table][pp_idx]

    def lookup_id(self, pgm_id: int) -> Dict[str, Union[str, int]]:
        """
        Retrieve the original name / key combination of a pgm object

        Args:
            pgm_id: a unique numerical ID

        Returns: The original table / index combination
        """
        for key, indices in self.idx_lookup.items():
            if pgm_id in indices:
                return {"table": key, "index": indices[pgm_id]}
        raise KeyError(pgm_id)
