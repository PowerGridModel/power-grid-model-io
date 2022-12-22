# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Panda Power Converter
"""
import re
from functools import lru_cache
from typing import Dict, Mapping, Optional, Tuple, Union

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

    __slots__ = ("_std_types", "pp_data", "pgm_data", "idx", "idx_lookup", "next_idx", "system_frequency")

    def __init__(self, std_types: Optional[StdTypes] = None, system_frequency: float = 50.0):
        """
        Prepare some member variables and optionally load "std_types"
        Args:
            std_types: standard type database of possible Line, Transformer and Three Winding Transformer types
            system_frequency: fundamental frequency of the alternating current and voltage in the Network measured in Hz
        """
        super().__init__(source=None, destination=None)
        self._std_types: StdTypes = std_types if std_types is not None else {}
        self.system_frequency: float = system_frequency
        self.pp_data: PandaPowerData = {}
        self.pgm_data: Dataset = {}
        self.idx: Dict[Tuple[str, Optional[str]], pd.Series] = {}
        self.idx_lookup: Dict[Tuple[str, Optional[str]], pd.Series] = {}
        self.next_idx = 0

    def _parse_data(
        self, data: PandaPowerData, data_type: str, extra_info: Optional[ExtraInfoLookup] = None
    ) -> Dataset:
        """
        Set up for conversion from PandaPower to power-grid-model
        Args:
            data: PandaPowerData, i.e. a dictionary with the components as keys and pd.DataFrames as values, with
            attribute names as columns and their values in the table
            data_type: power-grid-model data type, i.e. "input" or "update"
            extra_info: an optional dictionary where extra component info (that can't be specified in
            power-grid-model data) can be specified

        Returns:
            Converted power-grid-model data
        """

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
            for (pp_table, name), indices in self.idx_lookup.items():
                for pgm_idx, pp_idx in zip(indices.index, indices):
                    if name:
                        extra_info[pgm_idx] = {"id_reference": {"table": pp_table, "name": name, "index": pp_idx}}
                    else:
                        extra_info[pgm_idx] = {"id_reference": {"table": pp_table, "index": pp_idx}}

        return self.pgm_data

    def _serialize_data(self, data: Dataset, extra_info: Optional[ExtraInfoLookup]) -> PandaPowerData:
        raise NotImplementedError()

    def _create_input_data(self):
        """
        Performs the conversion from PandaPower to power-grid-model by calling individual conversion functions
        """
        self._create_pgm_input_nodes()
        self._create_pgm_input_lines()
        self._create_pgm_input_sources()
        self._create_pgm_input_sym_loads()
        self._create_pgm_input_shunts()
        self._create_pgm_input_transformers()
        self._create_pgm_input_sym_gens()
        self._create_pgm_input_three_winding_transformers()
        self._create_pgm_input_links()
        self._create_pgm_input_ward()
        self._create_pgm_input_xward()
        self._create_pgm_input_motor()

    def _create_pgm_input_nodes(self):
        """
        This function converts a Bus Dataframe of PandaPower to a power-grid-model Node input array.

        Returns:
            returns a power-grid-model structured array for the Node component
        """
        assert "node" not in self.pgm_data

        pp_busses = self.pp_data["bus"]

        if pp_busses.empty:
            return

        pgm_nodes = initialize_array(data_type="input", component_type="node", shape=len(pp_busses))
        pgm_nodes["id"] = self._generate_ids("bus", pp_busses.index)
        pgm_nodes["u_rated"] = self._get_pp_attr("bus", "vn_kv") * 1e3

        self.pgm_data["node"] = pgm_nodes

    def _create_pgm_input_lines(self):
        """
        This function converts a Line Dataframe of PandaPower to a power-grid-model Line input array.

        Returns:
            returns a power-grid-model structured array for the Line component
        """
        assert "line" not in self.pgm_data

        pp_lines = self.pp_data["line"]

        if pp_lines.empty:
            return

        switch_states = self.get_switch_states("line")

        pgm_lines = initialize_array(data_type="input", component_type="line", shape=len(pp_lines))
        pgm_lines["id"] = self._generate_ids("line", pp_lines.index)
        pgm_lines["from_node"] = self._get_ids("bus", pp_lines["from_bus"])
        pgm_lines["from_status"] = self._get_pp_attr("line", "in_service") & switch_states.iloc[0, :]
        pgm_lines["to_node"] = self._get_ids("bus", pp_lines["to_bus"])
        pgm_lines["to_status"] = self._get_pp_attr("line", "in_service") & switch_states.iloc[1, :]
        pgm_lines["r1"] = (
            self._get_pp_attr("line", "r_ohm_per_km")
            * self._get_pp_attr("line", "length_km")
            / self._get_pp_attr("line", "parallel")
        )
        pgm_lines["x1"] = (
            self._get_pp_attr("line", "x_ohm_per_km")
            * self._get_pp_attr("line", "length_km")
            / self._get_pp_attr("line", "parallel")
        )
        pgm_lines["c1"] = (
            self._get_pp_attr("line", "c_nf_per_km")
            * self._get_pp_attr("line", "length_km")
            * self._get_pp_attr("line", "parallel")
            * 1e-9
        )
        # The formula for tan1 = R_1 / Xc_1 = (g * 1e-6) / (2 * pi * f * c * 1e-9) = g / (2 * pi * f * c * 1e-3)
        pgm_lines["tan1"] = (
            self._get_pp_attr("line", "g_us_per_km")
            / self._get_pp_attr("line", "c_nf_per_km")
            / (2 * np.pi * self.system_frequency * 1e-3)
        )
        pgm_lines["i_n"] = (
            (self._get_pp_attr("line", "max_i_ka") * 1e3)
            * self._get_pp_attr("line", "df")
            * self._get_pp_attr("line", "parallel")
        )

        self.pgm_data["line"] = pgm_lines

    def _create_pgm_input_sources(self):
        """
        This function converts External Grid Dataframe of PandaPower to a power-grid-model Source input array.

        Returns:
            returns a power-grid-model structured array for the Source component
        """
        assert "source" not in self.pgm_data

        pp_ext_grid = self.pp_data["ext_grid"]

        if pp_ext_grid.empty:
            return

        pgm_sources = initialize_array(data_type="input", component_type="source", shape=len(pp_ext_grid))
        pgm_sources["id"] = self._generate_ids("ext_grid", pp_ext_grid.index)
        pgm_sources["node"] = self._get_ids("bus", pp_ext_grid["bus"])
        pgm_sources["status"] = self._get_pp_attr("ext_grid", "in_service")
        pgm_sources["u_ref"] = self._get_pp_attr("ext_grid", "vm_pu")
        pgm_sources["rx_ratio"] = self._get_pp_attr("ext_grid", "rx_max")
        pgm_sources["u_ref_angle"] = self._get_pp_attr("ext_grid", "va_degree") * (np.pi / 180)
        pgm_sources["sk"] = self._get_pp_attr("ext_grid", "s_sc_max_mva", np.nan) * 1e6

        self.pgm_data["source"] = pgm_sources

    def _create_pgm_input_shunts(self):
        """
        This function converts a Shunt Dataframe of PandaPower to a power-grid-model Shunt input array.

        Returns:
            returns a power-grid-model structured array for the Shunt component
        """
        assert "shunt" not in self.pgm_data

        pp_shunts = self.pp_data["shunt"]

        if pp_shunts.empty:
            return

        vn_kv_2 = self._get_pp_attr("shunt", "vn_kv") * self._get_pp_attr("shunt", "vn_kv")

        pgm_shunts = initialize_array(data_type="input", component_type="shunt", shape=len(pp_shunts))
        pgm_shunts["id"] = self._generate_ids("shunt", pp_shunts.index)
        pgm_shunts["node"] = self._get_ids("bus", pp_shunts["bus"])
        pgm_shunts["status"] = self._get_pp_attr("shunt", "in_service")
        pgm_shunts["g1"] = self._get_pp_attr("shunt", "p_mw") * self._get_pp_attr("shunt", "step") / vn_kv_2
        pgm_shunts["b1"] = -(self._get_pp_attr("shunt", "q_mvar") * self._get_pp_attr("shunt", "step")) / vn_kv_2

        self.pgm_data["shunt"] = pgm_shunts

    def _create_pgm_input_sym_gens(self):
        """
        This function converts a Static Generator Dataframe of PandaPower to a power-grid-model
        Symmetrical Generator input array.

        Returns:
            returns a power-grid-model structured array for the Symmetrical Generator component
        """
        assert "sym_gen" not in self.pgm_data

        pp_sgens = self.pp_data["sgen"]

        if pp_sgens.empty:
            return

        pgm_sym_gens = initialize_array(data_type="input", component_type="sym_gen", shape=len(pp_sgens))
        pgm_sym_gens["id"] = self._generate_ids("sgen", pp_sgens.index)
        pgm_sym_gens["node"] = self._get_ids("bus", pp_sgens["bus"])
        pgm_sym_gens["status"] = self._get_pp_attr("sgen", "in_service")
        pgm_sym_gens["p_specified"] = self._get_pp_attr("sgen", "p_mw") * 1e6 * self._get_pp_attr("sgen", "scaling")
        pgm_sym_gens["q_specified"] = self._get_pp_attr("sgen", "q_mvar") * 1e6 * self._get_pp_attr("sgen", "scaling")
        pgm_sym_gens["type"] = LoadGenType.const_power

        self.pgm_data["sym_gen"] = pgm_sym_gens

    def _create_pgm_input_sym_loads(self):
        """
        This function converts a Load Dataframe of PandaPower to a power-grid-model
        Symmetrical Load input array. For one load in PandaPower there are three loads in
        power-grid-model created.

        Returns:
            returns a power-grid-model structured array for the Symmetrical Load component
        """
        assert "sym_load" not in self.pgm_data

        pp_loads = self.pp_data["load"]

        if pp_loads.empty:
            return

        n_loads = len(pp_loads)

        pgm_sym_loads = initialize_array(data_type="input", component_type="sym_load", shape=3 * n_loads)

        const_i_multiplier = (
            self._get_pp_attr("load", "const_i_percent") * self._get_pp_attr("load", "scaling") * (1e-2 * 1e6)
        )
        const_z_multiplier = (
            self._get_pp_attr("load", "const_z_percent") * self._get_pp_attr("load", "scaling") * (1e-2 * 1e6)
        )
        const_p_multiplier = (1e6 - const_i_multiplier - const_z_multiplier) * pp_loads["scaling"]

        pgm_sym_loads["id"][:n_loads] = self._generate_ids("load", pp_loads.index, name="const_power")
        pgm_sym_loads["node"][:n_loads] = self._get_ids("bus", pp_loads["bus"])
        pgm_sym_loads["status"][:n_loads] = self._get_pp_attr("load", "in_service")
        pgm_sym_loads["type"][:n_loads] = LoadGenType.const_power
        pgm_sym_loads["p_specified"][:n_loads] = const_p_multiplier * self._get_pp_attr("load", "p_mw")
        pgm_sym_loads["q_specified"][:n_loads] = const_p_multiplier * self._get_pp_attr("load", "q_mvar")

        pgm_sym_loads["id"][n_loads : 2 * n_loads] = self._generate_ids("load", pp_loads.index, name="const_impedance")
        pgm_sym_loads["node"][n_loads : 2 * n_loads] = self._get_ids("bus", pp_loads["bus"])
        pgm_sym_loads["status"][n_loads : 2 * n_loads] = self._get_pp_attr("load", "in_service")
        pgm_sym_loads["type"][n_loads : 2 * n_loads] = LoadGenType.const_impedance
        pgm_sym_loads["p_specified"][n_loads : 2 * n_loads] = const_z_multiplier * self._get_pp_attr("load", "p_mw")
        pgm_sym_loads["q_specified"][n_loads : 2 * n_loads] = const_z_multiplier * self._get_pp_attr("load", "q_mvar")

        pgm_sym_loads["id"][-n_loads:] = self._generate_ids("load", pp_loads.index, name="const_current")
        pgm_sym_loads["node"][-n_loads:] = self._get_ids("bus", pp_loads["bus"])
        pgm_sym_loads["status"][-n_loads:] = self._get_pp_attr("load", "in_service")
        pgm_sym_loads["type"][-n_loads:] = LoadGenType.const_current
        pgm_sym_loads["p_specified"][-n_loads:] = const_i_multiplier * self._get_pp_attr("load", "p_mw")
        pgm_sym_loads["q_specified"][-n_loads:] = const_i_multiplier * self._get_pp_attr("load", "q_mvar")

        self.pgm_data["sym_load"] = pgm_sym_loads

    def _create_pgm_input_transformers(self):
        """
        This function converts a Transformer Dataframe of PandaPower to a power-grid-model
        Transformer input array.

        Returns:
            returns a power-grid-model structured array for the Transformer component
        """
        assert "transformer" not in self.pgm_data

        pp_trafo = self.pp_data["trafo"]

        if pp_trafo.empty:
            return

        switch_states = self.get_switch_states("trafo")
        winding_types = self.get_trafo_winding_types()

        pgm_transformers = initialize_array(data_type="input", component_type="transformer", shape=len(pp_trafo))
        pgm_transformers["id"] = self._generate_ids("trafo", pp_trafo.index)
        pgm_transformers["from_node"] = self._get_ids("bus", pp_trafo["hv_bus"])
        pgm_transformers["from_status"] = self._get_pp_attr("trafo", "in_service") & switch_states.iloc[0, :]
        pgm_transformers["to_node"] = self._get_ids("bus", pp_trafo["lv_bus"])
        pgm_transformers["to_status"] = self._get_pp_attr("trafo", "in_service") & switch_states.iloc[1, :]
        pgm_transformers["u1"] = self._get_pp_attr("trafo", "vn_hv_kv") * 1e3
        pgm_transformers["u2"] = self._get_pp_attr("trafo", "vn_lv_kv") * 1e3
        pgm_transformers["sn"] = self._get_pp_attr("trafo", "sn_mva") * self._get_pp_attr("trafo", "parallel") * 1e6
        pgm_transformers["uk"] = self._get_pp_attr("trafo", "vk_percent") * 1e-2
        pgm_transformers["pk"] = (
            self._get_pp_attr("trafo", "vkr_percent")
            * self._get_pp_attr("trafo", "sn_mva")
            * self._get_pp_attr("trafo", "parallel")
            * (1e6 * 1e-2)
        )
        pgm_transformers["i0"] = self._get_pp_attr("trafo", "i0_percent") * 1e-2
        pgm_transformers["p0"] = self._get_pp_attr("trafo", "pfe_kw") * self._get_pp_attr("trafo", "parallel") * 1e3
        pgm_transformers["winding_from"] = winding_types["winding_from"]
        pgm_transformers["winding_to"] = winding_types["winding_to"]
        pgm_transformers["clock"] = round(self._get_pp_attr("trafo", "shift_degree") / 30) % 12
        pgm_transformers["tap_pos"] = self._get_pp_attr("trafo", "tap_pos")
        pgm_transformers["tap_side"] = self._get_transformer_tap_side(pp_trafo["tap_side"])
        pgm_transformers["tap_min"] = self._get_pp_attr("trafo", "tap_min")
        pgm_transformers["tap_max"] = self._get_pp_attr("trafo", "tap_max")
        pgm_transformers["tap_nom"] = self._get_pp_attr("trafo", "tap_neutral")
        pgm_transformers["tap_size"] = self._get_tap_size(pp_trafo)

        self.pgm_data["transformer"] = pgm_transformers

    def _create_pgm_input_three_winding_transformers(self):
        """
        This function converts a Three Winding Transformer Dataframe of PandaPower to a power-grid-model
        Three Winding Transformer input array.

        Returns:
            returns a power-grid-model structured array for the Three Winding Transformer component
        """
        assert "three_winding_transformer" not in self.pgm_data

        pp_trafo3w = self.pp_data["trafo3w"]

        if pp_trafo3w.empty:
            return

        sn_hv_mva = self._get_pp_attr("trafo3w", "sn_hv_mva")
        sn_mv_mva = self._get_pp_attr("trafo3w", "sn_mv_mva")
        sn_lv_mva = self._get_pp_attr("trafo3w", "sn_lv_mva")

        switch_states = self.get_trafo3w_switch_states(pp_trafo3w)
        winding_type = self.get_trafo3w_winding_types()

        pgm_3wtransformers = initialize_array(
            data_type="input", component_type="three_winding_transformer", shape=len(pp_trafo3w)
        )
        pgm_3wtransformers["id"] = self._generate_ids("trafo3w", pp_trafo3w.index)
        pgm_3wtransformers["node_1"] = self._get_ids("bus", pp_trafo3w["hv_bus"])
        pgm_3wtransformers["node_2"] = self._get_ids("bus", pp_trafo3w["mv_bus"])
        pgm_3wtransformers["node_3"] = self._get_ids("bus", pp_trafo3w["lv_bus"])
        pgm_3wtransformers["status_1"] = self._get_pp_attr("trafo3w", "in_service") & switch_states.iloc[0, :]
        pgm_3wtransformers["status_2"] = self._get_pp_attr("trafo3w", "in_service") & switch_states.iloc[1, :]
        pgm_3wtransformers["status_3"] = self._get_pp_attr("trafo3w", "in_service") & switch_states.iloc[2, :]
        pgm_3wtransformers["u1"] = self._get_pp_attr("trafo3w", "vn_hv_kv") * 1e3
        pgm_3wtransformers["u2"] = self._get_pp_attr("trafo3w", "vn_mv_kv") * 1e3
        pgm_3wtransformers["u3"] = self._get_pp_attr("trafo3w", "vn_lv_kv") * 1e3
        pgm_3wtransformers["sn_1"] = self._get_pp_attr("trafo3w", "sn_hv_mva") * 1e6
        pgm_3wtransformers["sn_2"] = self._get_pp_attr("trafo3w", "sn_mv_mva") * 1e6
        pgm_3wtransformers["sn_3"] = self._get_pp_attr("trafo3w", "sn_lv_mva") * 1e6
        pgm_3wtransformers["uk_12"] = self._get_pp_attr("trafo3w", "vk_hv_percent") * 1e-2
        pgm_3wtransformers["uk_13"] = self._get_pp_attr("trafo3w", "vk_lv_percent") * 1e-2
        pgm_3wtransformers["uk_23"] = self._get_pp_attr("trafo3w", "vk_mv_percent") * 1e-2

        pgm_3wtransformers["pk_12"] = (
            self._get_pp_attr("trafo3w", "vkr_hv_percent") * np.minimum(sn_hv_mva, sn_mv_mva) * (1e-2 * 1e6)
        )

        pgm_3wtransformers["pk_13"] = (
            self._get_pp_attr("trafo3w", "vkr_lv_percent") * np.minimum(sn_hv_mva, sn_lv_mva) * (1e-2 * 1e6)
        )

        pgm_3wtransformers["pk_23"] = (
            self._get_pp_attr("trafo3w", "vkr_mv_percent") * np.minimum(sn_mv_mva, sn_lv_mva) * (1e-2 * 1e6)
        )

        pgm_3wtransformers["i0"] = self._get_pp_attr("trafo3w", "i0_percent") * 1e-2
        pgm_3wtransformers["p0"] = self._get_pp_attr("trafo3w", "pfe_kw") * 1e3
        pgm_3wtransformers["winding_1"] = winding_type["winding_1"]
        pgm_3wtransformers["winding_2"] = winding_type["winding_2"]
        pgm_3wtransformers["winding_3"] = winding_type["winding_3"]
        pgm_3wtransformers["clock_12"] = round(self._get_pp_attr("trafo3w", "shift_mv_degree") / 30.0) % 12
        pgm_3wtransformers["clock_13"] = round(self._get_pp_attr("trafo3w", "shift_lv_degree") / 30.0) % 12
        pgm_3wtransformers["tap_pos"] = self._get_pp_attr("trafo3w", "tap_pos")
        pgm_3wtransformers["tap_side"] = self._get_3wtransformer_tap_side(
            pd.Series(self._get_pp_attr("trafo3w", "tap_side"))
        )
        pgm_3wtransformers["tap_min"] = self._get_pp_attr("trafo3w", "tap_min")
        pgm_3wtransformers["tap_max"] = self._get_pp_attr("trafo3w", "tap_max")
        pgm_3wtransformers["tap_nom"] = self._get_pp_attr("trafo3w", "tap_neutral")
        pgm_3wtransformers["tap_size"] = self._get_3wtransformer_tap_size(pp_trafo3w)

        self.pgm_data["three_winding_transformer"] = pgm_3wtransformers

    def _create_pgm_input_links(self):
        """
        This function takes a Switch Dataframe of PandaPower, extracts the Switches which have Bus to Bus
        connection and converts them to a power-grid-model Link input array.

        Returns:
            returns a power-grid-model structured array for the Link component
        """
        assert "link" not in self.pgm_data

        pp_switches = self.pp_data["switch"]

        if pp_switches.empty:
            return

        pp_switches = pp_switches[
            self._get_pp_attr("switch", "et") == "b"
        ]  # This should take all the switches which are b2b

        self.pp_data["switch_b2b"] = pp_switches  # Create a table in pp_data for bus to bus switches and then access
        # it to get the closed attribute. We do this so that we could later easily get the closed attribute,
        # if we don't do this the attribute closed will be taken from all the switches, rather than from only bus to
        # bus, that will result in an error

        pgm_links = initialize_array(data_type="input", component_type="link", shape=len(pp_switches))
        pgm_links["id"] = self._generate_ids("switch", pp_switches.index, name="bus_to_bus")
        pgm_links["from_node"] = self._get_ids("bus", pp_switches["bus"])
        pgm_links["to_node"] = self._get_ids("bus", pp_switches["element"])
        pgm_links["from_status"] = self._get_pp_attr("switch_b2b", "closed")
        pgm_links["to_status"] = self._get_pp_attr("switch_b2b", "closed")

        self.pgm_data["link"] = pgm_links

    def _create_pgm_input_ward(self):  # pragma: no cover
        # TODO: create unit tests for the function
        pp_wards = self.pp_data["ward"]

        if pp_wards.empty:
            return

        raise NotImplementedError("Ward is not implemented yet!")

    def _create_pgm_input_xward(self):  # pragma: no cover
        # TODO: create unit tests for the function
        pp_xwards = self.pp_data["xward"]

        if pp_xwards.empty:
            return

        raise NotImplementedError("Extended Ward is not implemented yet!")

    def _create_pgm_input_motor(self):  # pragma: no cover
        # TODO: create unit tests for the function
        pp_motors = self.pp_data["motor"]

        if pp_motors.empty:
            return

        raise NotImplementedError("Motor is not implemented yet!")

    def _generate_ids(self, pp_table: str, pp_idx: pd.Index, name: Optional[str] = None) -> np.arange:
        """
        Generate numerical power-grid-model IDs for a PandaPower component

        Args:
            pp_table: Table name (e.g. "bus")
            pp_idx: PandaPower component identifier

        Returns:
            The generated IDs
        """
        key = (pp_table, name)
        assert key not in self.idx_lookup
        n_objects = len(pp_idx)
        pgm_idx = np.arange(start=self.next_idx, stop=self.next_idx + n_objects, dtype=np.int32)
        self.idx[key] = pd.Series(pgm_idx, index=pp_idx)
        self.idx_lookup[key] = pd.Series(pp_idx, index=pgm_idx)
        self.next_idx += n_objects
        return pgm_idx

    def _get_ids(self, pp_table: str, pp_idx: pd.Series, name: Optional[str] = None) -> pd.Series:
        """
        Get numerical power-grid-model IDs for a PandaPower component

        Args:
            pp_table: Table name (e.g. "bus")
            pp_idx: PandaPower component identifier

        Returns:
            The IDs if they were previously generated
        """
        key = (pp_table, name)
        if key not in self.idx:
            raise KeyError(f"No indexes have been created for '{pp_table}' (name={name})!")
        return self.idx[key][pp_idx]

    @staticmethod
    def _get_tap_size(pp_trafo: pd.DataFrame) -> np.ndarray:
        """
        Calculate the "tap size" of Transformers

        Args:
            pp_trafo: PandaPower dataframe with information about the Transformers in
            the Network (e.g. "hv_bus", "i0_percent")

        Returns:
            The "tap size" of Transformers
        """
        tap_side_hv = np.array(pp_trafo["tap_side"] == "hv")
        tap_side_lv = np.array(pp_trafo["tap_side"] == "lv")
        tap_step_multiplier = pp_trafo["tap_step_percent"] * (1e-2 * 1e3)

        tap_size = np.empty(shape=len(pp_trafo), dtype=np.float64)
        tap_size[tap_side_hv] = tap_step_multiplier[tap_side_hv] * pp_trafo["vn_hv_kv"][tap_side_hv]
        tap_size[tap_side_lv] = tap_step_multiplier[tap_side_lv] * pp_trafo["vn_lv_kv"][tap_side_lv]

        return tap_size

    @staticmethod
    def _get_transformer_tap_side(tap_side: pd.Series) -> np.ndarray:
        """
        Get the enumerated "tap side" of Transformers

        Args:
            tap_side: PandaPower series with information about the "tap_side" attribute

        Returns:
            The enumerated "tap side"
        """
        new_tap_side = np.array(tap_side)
        new_tap_side[new_tap_side == "hv"] = BranchSide.from_side
        new_tap_side[new_tap_side == "lv"] = BranchSide.to_side

        return new_tap_side

    @staticmethod
    def _get_3wtransformer_tap_side(tap_side: pd.Series) -> np.ndarray:
        """
        Get the enumerated "tap side" of Three Winding Transformers

        Args:
            tap_side: PandaPower series with information about the "tap_side" attribute

        Returns:
            The enumerated "tap side"
        """
        new_tap_side = np.array(tap_side)
        new_tap_side[new_tap_side == "hv"] = Branch3Side.side_1
        new_tap_side[new_tap_side == "mv"] = Branch3Side.side_2
        new_tap_side[new_tap_side == "lv"] = Branch3Side.side_3

        return new_tap_side

    @staticmethod
    def _get_3wtransformer_tap_size(pp_3wtrafo: pd.DataFrame) -> np.ndarray:
        """
        Calculate the "tap size" of Three Winding Transformers

        Args:
            pp_3wtrafo: PandaPower dataframe with information about the Three Winding Transformers in
            the Network (e.g. "hv_bus", "i0_percent")

        Returns:
            The "tap size" of Three Winding Transformers
        """
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
        Get the state of an individual switch. Can be open or closed.

        Args:
            component: PandaPower dataframe with information about the component that is connected to the switch.
            Can be a Line dataframe, Transformer dataframe or Three Winding Transformer dataframe.

            switches: PandaPower dataframe with information about the switches, has
            such attributes as: "element", "bus", "closed"

            bus: name of the bus attribute that the component connects to (e.g "hv_bus", "from_bus", "lv_bus", etc.)

        Returns:
            The "closed" value of a Switch
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
        Return switch states of either Lines or Transformers

        Args:
            pp_table: Table name (e.g. "bus")

        Returns:
            The switch states of either Lines or Transformers
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
        pp_switches = pp_switches[pp_switches["et"] == element_type]
        pp_switches = pp_switches[["element", "bus", "closed"]]

        pp_from_switches = self.get_individual_switch_states(component, pp_switches, bus1)
        pp_to_switches = self.get_individual_switch_states(component, pp_switches, bus2)

        return pd.DataFrame(data=(pp_from_switches, pp_to_switches))

    def get_trafo3w_switch_states(self, trafo3w: pd.DataFrame) -> pd.DataFrame:
        """
        Return switch states of Three Winding Transformers

        Args:
            trafo3w: PandaPower dataframe with information about the Three Winding Transformers.

        Returns:
            The switch states of Three Winding Transformers
        """
        element_type = "t3"
        bus1 = "hv_bus"
        bus2 = "mv_bus"
        bus3 = "lv_bus"
        trafo3w["index"] = trafo3w.index

        # Select the appropriate switches and columns
        pp_switches = self.pp_data["switch"]
        pp_switches = pp_switches[pp_switches["et"] == element_type]
        pp_switches = pp_switches[["element", "bus", "closed"]]

        # Join the switches with the three winding trafo three times, for the hv_bus, mv_bus and once for the lv_bus
        pp_1_switches = self.get_individual_switch_states(trafo3w, pp_switches, bus1)
        pp_2_switches = self.get_individual_switch_states(trafo3w, pp_switches, bus2)
        pp_3_switches = self.get_individual_switch_states(trafo3w, pp_switches, bus3)

        return pd.DataFrame((pp_1_switches, pp_2_switches, pp_3_switches))

    def get_trafo_winding_types(self) -> pd.DataFrame:
        """
        This function extracts Transformers' "winding_type" attribute through "vector_group" attribute or
        through "std_type" attribute.

        Returns:
            returns the "from" and "to" winding types of a transformer
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
            return vector_group_to_winding_types(self._std_types["trafo"][std_type]["vector_group"])

        trafo = self.pp_data["trafo"]
        if "vector_group" in trafo:
            trafo = trafo["vector_group"].apply(vector_group_to_winding_types)
        else:
            trafo = trafo["std_type"].apply(std_type_to_winding_types)
        trafo.columns = ["winding_from", "winding_to"]
        return trafo

    def get_trafo3w_winding_types(self) -> pd.DataFrame:
        """
        This function extracts Three Winding Transformers' "winding_type" attribute through "vector_group" attribute or
        through "std_type" attribute.

        Returns:
            returns the three winding types of Three Winding Transformers
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
            return vector_group_to_winding_types(self._std_types["trafo3w"][std_type]["vector_group"])

        trafo3w = self.pp_data["trafo3w"]
        if "vector_group" in trafo3w:
            trafo3w = trafo3w["vector_group"].apply(vector_group_to_winding_types)
        else:
            trafo3w = trafo3w["std_type"].apply(std_type_to_winding_types)
        trafo3w.columns = ["winding_1", "winding_2", "winding_3"]
        return trafo3w

    def _get_pp_attr(self, table: str, attribute: str, default: Optional[float] = None) -> Union[np.ndarray, float]:
        """
        Returns the selected PandaPower attribute from the selected PandaPower table.

        Args:
            table: Table name (e.g. "bus")
            attribute: an attribute from the table (e.g "vn_kv")

        Returns:
            Returns the selected PandaPower attribute from the selected PandaPower table
        """
        pp_component_data = self.pp_data[table]

        # If the attribute exists, return it
        if attribute in pp_component_data:
            return pp_component_data[attribute]

        # Try to find the std_type value for this attribute
        if self._std_types is not None and table in self._std_types and "std_type" in pp_component_data:
            std_types = self._std_types[table]

            @lru_cache
            def get_std_value(std_type_name: str):
                std_type = std_types[std_type_name]
                if attribute in std_type:
                    return std_type[attribute]
                if default is not None:
                    return default
                raise KeyError(f"No '{attribute}' value for '{table}' with std_type '{std_type_name}'.")

            return pp_component_data["std_type"].apply(get_std_value)

        # Return the default value (assume that broadcasting is handled by the caller / numpy)
        if default is None:
            raise KeyError(f"No '{attribute}' value for '{table}'.")
        return default

    def get_id(self, pp_table: str, pp_idx: int, name: Optional[str] = None) -> int:
        """
        Get a numerical ID previously associated with the supplied table / index combination

        Args:
            pp_table: Table name (e.g. "bus")
            pp_idx: PandaPower component identifier

        Returns:
            The associated id
        """
        return self.idx[(pp_table, name)][pp_idx]

    def lookup_id(self, pgm_id: int) -> Dict[str, Union[str, int]]:
        """
        Retrieve the original name / key combination of a pgm object

        Args:
            pgm_id: a unique numerical ID

        Returns:
            The original table / index combination
        """
        for (table, name), indices in self.idx_lookup.items():
            if pgm_id in indices:
                if name:
                    return {"table": table, "name": name, "index": indices[pgm_id]}
                return {"table": table, "index": indices[pgm_id]}
        raise KeyError(pgm_id)
