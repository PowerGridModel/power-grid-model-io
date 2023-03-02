# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
# pylint: disable = too-many-lines
"""
Panda Power Converter
"""
from functools import lru_cache
from typing import Dict, List, MutableMapping, Optional, Tuple, Union

import numpy as np
import pandas as pd
from power_grid_model import Branch3Side, BranchSide, LoadGenType, WindingType, initialize_array, power_grid_meta_data
from power_grid_model.data_types import Dataset, SingleDataset

from power_grid_model_io.converters.base_converter import BaseConverter
from power_grid_model_io.data_types import ExtraInfoLookup
from power_grid_model_io.functions import get_winding
from power_grid_model_io.utils.regex import NODE_REF_RE, TRAFO3_CONNECTION_RE, TRAFO_CONNECTION_RE

PandaPowerData = MutableMapping[str, pd.DataFrame]


# pylint: disable=too-many-instance-attributes
class PandaPowerConverter(BaseConverter[PandaPowerData]):
    """
    Panda Power Converter
    """

    __slots__ = ("pp_input_data", "pgm_input_data", "idx", "idx_lookup", "next_idx", "system_frequency")

    def __init__(self, system_frequency: float = 50.0, trafo_loading: str = "current"):
        """
        Prepare some member variables

        Args:
            system_frequency: fundamental frequency of the alternating current and voltage in the Network measured in Hz
        """
        super().__init__(source=None, destination=None)
        self.trafo_loading = trafo_loading
        self.system_frequency: float = system_frequency
        self.pp_input_data: PandaPowerData = {}
        self.pgm_input_data: SingleDataset = {}
        self.pp_output_data: PandaPowerData = {}
        self.pgm_output_data: SingleDataset = {}
        self.pgm_nodes_lookup: pd.DataFrame = pd.DataFrame()
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
        self.pgm_input_data = {}
        self.idx_lookup = {}
        self.next_idx = 0

        # Set pandas data
        self.pp_input_data = data

        # Convert
        if data_type == "input":
            self._create_input_data()
        else:
            raise ValueError(f"Data type: '{data_type}' is not implemented")

        # Construct extra_info
        if extra_info is not None:
            self._fill_extra_info(extra_info=extra_info)

        return self.pgm_input_data

    def _serialize_data(self, data: Dataset, extra_info: Optional[ExtraInfoLookup]) -> PandaPowerData:
        """
        Set up for conversion from power-grid-model to PandaPower

        Args:
            data: a structured array of power-grid-model data.
            extra_info: an optional dictionary where extra component info (that can't be specified in
            power-grid-model data) can be specified

        Returns:
            Converted PandaPower data
        """

        # Clear pp data
        self.pgm_nodes_lookup = pd.DataFrame()
        self.pp_output_data = {}

        self.pgm_output_data = data

        # If extra_info is supplied, index lookups and node lookups should be created accordingly
        if extra_info is not None:
            self._extra_info_to_idx_lookup(extra_info)
            self._extra_info_to_pgm_input_data(extra_info)

        # Convert
        self._create_output_data()

        return self.pp_output_data

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
        self._create_pgm_input_asym_loads()
        self._create_pgm_input_asym_gens()
        self._create_pgm_input_wards()
        self._create_pgm_input_motors()
        self._create_pgm_input_storages()
        self._create_pgm_input_impedances()
        self._create_pgm_input_xwards()
        self._create_pgm_input_generators()
        self._create_pgm_input_dclines()

    def _fill_extra_info(self, extra_info: ExtraInfoLookup):
        for (pp_table, name), indices in self.idx_lookup.items():
            for pgm_id, pp_idx in zip(indices.index, indices):
                if name:
                    extra_info[pgm_id] = {"id_reference": {"table": pp_table, "name": name, "index": pp_idx}}
                else:
                    extra_info[pgm_id] = {"id_reference": {"table": pp_table, "index": pp_idx}}
        for component_data in self.pgm_input_data.values():
            for attr_name in component_data.dtype.names:
                if NODE_REF_RE.fullmatch(attr_name):
                    for pgm_id, node_id in component_data[["id", attr_name]]:
                        if pgm_id not in extra_info:
                            extra_info[pgm_id] = {attr_name: node_id}
                        else:
                            extra_info[pgm_id][attr_name] = node_id

    def _extra_info_to_idx_lookup(self, extra_info: ExtraInfoLookup):
        """
        Converts extra component info into idx_lookup

        Args:
            extra_info: a dictionary where the original panda power ids are stored
        """
        self.idx = {}
        self.idx_lookup = {}
        pgm_to_pp_id: Dict[Tuple[str, Optional[str]], List[Tuple[int, int]]] = {}
        for pgm_idx, extra in extra_info.items():
            if "id_reference" not in extra:
                continue
            assert isinstance(extra["id_reference"], dict)
            pp_table = extra["id_reference"]["table"]
            pp_index = extra["id_reference"]["index"]
            pp_name = extra["id_reference"].get("name")
            key = (pp_table, pp_name)
            if key not in pgm_to_pp_id:
                pgm_to_pp_id[key] = []
            pgm_to_pp_id[key].append((pgm_idx, pp_index))
        for key, table_pgm_to_pp_id in pgm_to_pp_id.items():
            pgm_ids, pp_indices = zip(*table_pgm_to_pp_id)
            self.idx[key] = pd.Series(pgm_ids, index=pp_indices)
            self.idx_lookup[key] = pd.Series(pp_indices, index=pgm_ids)

    def _extra_info_to_pgm_input_data(self, extra_info: ExtraInfoLookup):
        """
        Converts extra component info into node_lookup

        Args:
            extra_info: a dictionary where the node reference ids are stored
        """
        assert not self.pgm_input_data
        assert self.pgm_output_data

        dtype = np.int32
        nan = np.iinfo(dtype).min
        for component, data in self.pgm_output_data.items():
            input_cols = power_grid_meta_data["input"][component].dtype.names
            node_cols = [col for col in input_cols if NODE_REF_RE.fullmatch(col)]
            if not node_cols:
                continue
            num_cols = 1 + len(node_cols)
            ref = np.full(
                shape=len(data),
                fill_value=nan,
                dtype={"names": ["id"] + node_cols, "formats": [dtype] * num_cols},
            )
            for i, pgm_id in enumerate(data["id"]):
                extra = extra_info[pgm_id]
                ref[i] = (pgm_id,) + tuple(extra[col] for col in node_cols)
            self.pgm_input_data[component] = ref

    def _create_output_data(self):
        """
        Performs the conversion from power-grid-model to PandaPower by calling individual conversion functions.
        Furthermore, creates a global node lookup table, which stores nodes' voltage magnitude per unit and the voltage
        angle in degrees
        """
        # Many pp components store the voltage magnitude per unit and the voltage angle in degrees,
        # so let's create a global lookup table (indexed on the pgm ids)
        self.pgm_nodes_lookup = pd.DataFrame(
            {
                "u_pu": self.pgm_output_data["node"]["u_pu"],
                "u_degree": self.pgm_output_data["node"]["u_angle"] * (180.0 / np.pi),
            },
            index=self.pgm_output_data["node"]["id"],
        )

        self._pp_buses_output()
        self._pp_lines_output()
        self._pp_ext_grids_output()
        self._pp_loads_output()
        self._pp_shunts_output()
        self._pp_trafos_output()
        self._pp_sgens_output()
        self._pp_trafos3w_output()
        self._pp_ward_output()
        self._pp_motor_output()
        self._pp_asym_gens_output()
        self._pp_asym_loads_output()
        # Switches derive results from branches pp_output_data and pgm_output_data of links. Hence, placed in the end.
        self._pp_switches_output()

    def _create_pgm_input_nodes(self):
        """
        This function converts a Bus Dataframe of PandaPower to a power-grid-model Node input array.

        Returns:
            a power-grid-model structured array for the Node component
        """
        pp_busses = self.pp_input_data["bus"]

        if pp_busses.empty:
            return

        pgm_nodes = initialize_array(data_type="input", component_type="node", shape=len(pp_busses))
        pgm_nodes["id"] = self._generate_ids("bus", pp_busses.index)
        pgm_nodes["u_rated"] = self._get_pp_attr("bus", "vn_kv") * 1e3

        assert "node" not in self.pgm_input_data
        self.pgm_input_data["node"] = pgm_nodes

    def _create_pgm_input_lines(self):
        """
        This function converts a Line Dataframe of PandaPower to a power-grid-model Line input array.

        Returns:
            a power-grid-model structured array for the Line component
        """
        pp_lines = self.pp_input_data["line"]

        if pp_lines.empty:
            return

        switch_states = self.get_switch_states("line")
        in_service = self._get_pp_attr("line", "in_service", True)
        length_km = self._get_pp_attr("line", "length_km")
        parallel = self._get_pp_attr("line", "parallel", 1)
        c_nf_per_km = self._get_pp_attr("line", "c_nf_per_km")
        multiplier = length_km / parallel

        pgm_lines = initialize_array(data_type="input", component_type="line", shape=len(pp_lines))
        pgm_lines["id"] = self._generate_ids("line", pp_lines.index)
        pgm_lines["from_node"] = self._get_pgm_ids("bus", self._get_pp_attr("line", "from_bus"))
        pgm_lines["from_status"] = in_service & switch_states["from"]
        pgm_lines["to_node"] = self._get_pgm_ids("bus", self._get_pp_attr("line", "to_bus"))
        pgm_lines["to_status"] = in_service & switch_states["to"]
        pgm_lines["r1"] = self._get_pp_attr("line", "r_ohm_per_km") * multiplier
        pgm_lines["x1"] = self._get_pp_attr("line", "x_ohm_per_km") * multiplier
        pgm_lines["c1"] = c_nf_per_km * length_km * parallel * 1e-9
        # The formula for tan1 = R_1 / Xc_1 = (g * 1e-6) / (2 * pi * f * c * 1e-9) = g / (2 * pi * f * c * 1e-3)
        pgm_lines["tan1"] = (
            self._get_pp_attr("line", "g_us_per_km", 0) / c_nf_per_km / (2 * np.pi * self.system_frequency * 1e-3)
        )
        pgm_lines["i_n"] = (self._get_pp_attr("line", "max_i_ka") * 1e3) * self._get_pp_attr("line", "df", 1) * parallel

        assert "line" not in self.pgm_input_data
        self.pgm_input_data["line"] = pgm_lines

    def _create_pgm_input_sources(self):
        """
        This function converts External Grid Dataframe of PandaPower to a power-grid-model Source input array.

        Returns:
            a power-grid-model structured array for the Source component
        """
        pp_ext_grid = self.pp_input_data["ext_grid"]

        if pp_ext_grid.empty:
            return

        pgm_sources = initialize_array(data_type="input", component_type="source", shape=len(pp_ext_grid))
        pgm_sources["id"] = self._generate_ids("ext_grid", pp_ext_grid.index)
        pgm_sources["node"] = self._get_pgm_ids("bus", self._get_pp_attr("ext_grid", "bus"))
        pgm_sources["status"] = self._get_pp_attr("ext_grid", "in_service", True)
        pgm_sources["u_ref"] = self._get_pp_attr("ext_grid", "vm_pu", 1.0)
        pgm_sources["rx_ratio"] = self._get_pp_attr("ext_grid", "rx_max", np.nan)
        pgm_sources["u_ref_angle"] = self._get_pp_attr("ext_grid", "va_degree", 0.0) * (np.pi / 180)
        pgm_sources["sk"] = self._get_pp_attr("ext_grid", "s_sc_max_mva", np.nan) * 1e6

        assert "source" not in self.pgm_input_data
        self.pgm_input_data["source"] = pgm_sources

    def _create_pgm_input_shunts(self):
        """
        This function converts a Shunt Dataframe of PandaPower to a power-grid-model Shunt input array.

        Returns:
            a power-grid-model structured array for the Shunt component
        """
        pp_shunts = self.pp_input_data["shunt"]

        if pp_shunts.empty:
            return

        vn_kv = self._get_pp_attr("shunt", "vn_kv")
        vn_kv_2 = vn_kv * vn_kv

        step = self._get_pp_attr("shunt", "step", 1)

        pgm_shunts = initialize_array(data_type="input", component_type="shunt", shape=len(pp_shunts))
        pgm_shunts["id"] = self._generate_ids("shunt", pp_shunts.index)
        pgm_shunts["node"] = self._get_pgm_ids("bus", self._get_pp_attr("shunt", "bus"))
        pgm_shunts["status"] = self._get_pp_attr("shunt", "in_service", 1)
        pgm_shunts["g1"] = self._get_pp_attr("shunt", "p_mw") * step / vn_kv_2
        pgm_shunts["b1"] = -self._get_pp_attr("shunt", "q_mvar") * step / vn_kv_2

        assert "shunt" not in self.pgm_input_data
        self.pgm_input_data["shunt"] = pgm_shunts

    def _create_pgm_input_sym_gens(self):
        """
        This function converts a Static Generator Dataframe of PandaPower to a power-grid-model
        Symmetrical Generator input array.

        Returns:
            a power-grid-model structured array for the Symmetrical Generator component
        """
        pp_sgens = self.pp_input_data["sgen"]

        if pp_sgens.empty:
            return

        scaling = self._get_pp_attr("sgen", "scaling", 1.0)

        pgm_sym_gens = initialize_array(data_type="input", component_type="sym_gen", shape=len(pp_sgens))
        pgm_sym_gens["id"] = self._generate_ids("sgen", pp_sgens.index)
        pgm_sym_gens["node"] = self._get_pgm_ids("bus", self._get_pp_attr("sgen", "bus"))
        pgm_sym_gens["status"] = self._get_pp_attr("sgen", "in_service", True)
        pgm_sym_gens["p_specified"] = self._get_pp_attr("sgen", "p_mw") * (1e6 * scaling)
        pgm_sym_gens["q_specified"] = self._get_pp_attr("sgen", "q_mvar", 0.0) * (1e6 * scaling)
        pgm_sym_gens["type"] = LoadGenType.const_power

        assert "sym_gen" not in self.pgm_input_data
        self.pgm_input_data["sym_gen"] = pgm_sym_gens

    def _create_pgm_input_asym_gens(self):
        """
        This function converts an Asymmetric Static Generator Dataframe of PandaPower to a power-grid-model
        Asymmetrical Generator input array.

        Returns:
            a power-grid-model structured array for the Asymmetrical Generator component
        """
        # TODO: create unit tests for asym_gen conversion
        pp_asym_gens = self.pp_input_data["asymmetric_sgen"]

        if pp_asym_gens.empty:
            return

        scaling = self._get_pp_attr("asymmetric_sgen", "scaling")
        multiplier = 1e6 * scaling

        pgm_asym_gens = initialize_array(data_type="input", component_type="asym_gen", shape=len(pp_asym_gens))
        pgm_asym_gens["id"] = self._generate_ids("asymmetric_sgen", pp_asym_gens.index)
        pgm_asym_gens["node"] = self._get_pgm_ids("bus", self._get_pp_attr("asymmetric_sgen", "bus"))
        pgm_asym_gens["status"] = self._get_pp_attr("asymmetric_sgen", "in_service")
        pgm_asym_gens["p_specified"] = np.transpose(
            np.array(
                (
                    self._get_pp_attr("asymmetric_sgen", "p_a_mw"),
                    self._get_pp_attr("asymmetric_sgen", "p_b_mw"),
                    self._get_pp_attr("asymmetric_sgen", "p_c_mw"),
                )
            )
            * multiplier
        )
        pgm_asym_gens["q_specified"] = np.transpose(
            np.array(
                (
                    self._get_pp_attr("asymmetric_sgen", "q_a_mvar"),
                    self._get_pp_attr("asymmetric_sgen", "q_b_mvar"),
                    self._get_pp_attr("asymmetric_sgen", "q_c_mvar"),
                )
            )
            * multiplier
        )
        pgm_asym_gens["type"] = LoadGenType.const_power

        assert "asym_gen" not in self.pgm_input_data
        self.pgm_input_data["asym_gen"] = pgm_asym_gens

    def _create_pgm_input_sym_loads(self):
        """
        This function converts a Load Dataframe of PandaPower to a power-grid-model
        Symmetrical Load input array. For one load in PandaPower there are three loads in
        power-grid-model created.

        Returns:
            a power-grid-model structured array for the Symmetrical Load component
        """
        pp_loads = self.pp_input_data["load"]

        if pp_loads.empty:
            return

        scaling = self._get_pp_attr("load", "scaling", 1.0)
        in_service = self._get_pp_attr("load", "in_service", True)
        p_mw = self._get_pp_attr("load", "p_mw", 0.0)
        q_mvar = self._get_pp_attr("load", "q_mvar", 0.0)
        bus = self._get_pp_attr("load", "bus")

        n_loads = len(pp_loads)

        pgm_sym_loads = initialize_array(data_type="input", component_type="sym_load", shape=3 * n_loads)

        const_i_multiplier = self._get_pp_attr("load", "const_i_percent", 0) * scaling * (1e-2 * 1e6)
        const_z_multiplier = self._get_pp_attr("load", "const_z_percent", 0) * scaling * (1e-2 * 1e6)
        const_p_multiplier = (1e6 - const_i_multiplier - const_z_multiplier) * scaling

        pgm_sym_loads["id"][:n_loads] = self._generate_ids("load", pp_loads.index, name="const_power")
        pgm_sym_loads["node"][:n_loads] = self._get_pgm_ids("bus", bus)
        pgm_sym_loads["status"][:n_loads] = in_service
        pgm_sym_loads["type"][:n_loads] = LoadGenType.const_power
        pgm_sym_loads["p_specified"][:n_loads] = const_p_multiplier * p_mw
        pgm_sym_loads["q_specified"][:n_loads] = const_p_multiplier * q_mvar

        pgm_sym_loads["id"][n_loads : 2 * n_loads] = self._generate_ids("load", pp_loads.index, name="const_impedance")
        pgm_sym_loads["node"][n_loads : 2 * n_loads] = self._get_pgm_ids("bus", bus)
        pgm_sym_loads["status"][n_loads : 2 * n_loads] = in_service
        pgm_sym_loads["type"][n_loads : 2 * n_loads] = LoadGenType.const_impedance
        pgm_sym_loads["p_specified"][n_loads : 2 * n_loads] = const_z_multiplier * p_mw
        pgm_sym_loads["q_specified"][n_loads : 2 * n_loads] = const_z_multiplier * q_mvar

        pgm_sym_loads["id"][-n_loads:] = self._generate_ids("load", pp_loads.index, name="const_current")
        pgm_sym_loads["node"][-n_loads:] = self._get_pgm_ids("bus", bus)
        pgm_sym_loads["status"][-n_loads:] = in_service
        pgm_sym_loads["type"][-n_loads:] = LoadGenType.const_current
        pgm_sym_loads["p_specified"][-n_loads:] = const_i_multiplier * p_mw
        pgm_sym_loads["q_specified"][-n_loads:] = const_i_multiplier * q_mvar

        assert "sym_load" not in self.pgm_input_data
        self.pgm_input_data["sym_load"] = pgm_sym_loads

    def _create_pgm_input_asym_loads(self):
        """
        This function converts an asymmetric_load Dataframe of PandaPower to a power-grid-model asym_load input array.

        Returns:
            a power-grid-model structured array for the asym_load component
        """
        # TODO: create unit tests for asym_load conversion
        pp_asym_loads = self.pp_input_data["asymmetric_load"]

        if pp_asym_loads.empty:
            return

        scaling = self._get_pp_attr("asymmetric_load", "scaling")
        multiplier = 1e6 * scaling

        pgm_asym_loads = initialize_array(data_type="input", component_type="asym_load", shape=len(pp_asym_loads))
        pgm_asym_loads["id"] = self._generate_ids("asymmetric_load", pp_asym_loads.index)
        pgm_asym_loads["node"] = self._get_pgm_ids("bus", self._get_pp_attr("asymmetric_load", "bus"))
        pgm_asym_loads["status"] = self._get_pp_attr("asymmetric_load", "in_service")
        pgm_asym_loads["p_specified"] = np.transpose(
            np.array(
                [
                    self._get_pp_attr("asymmetric_load", "p_a_mw"),
                    self._get_pp_attr("asymmetric_load", "p_b_mw"),
                    self._get_pp_attr("asymmetric_load", "p_c_mw"),
                ]
            )
            * multiplier
        )
        pgm_asym_loads["q_specified"] = np.transpose(
            np.array(
                [
                    self._get_pp_attr("asymmetric_load", "q_a_mvar"),
                    self._get_pp_attr("asymmetric_load", "q_b_mvar"),
                    self._get_pp_attr("asymmetric_load", "q_c_mvar"),
                ]
            )
            * multiplier
        )
        pgm_asym_loads["type"] = LoadGenType.const_power

        assert "asym_load" not in self.pgm_input_data
        self.pgm_input_data["asym_load"] = pgm_asym_loads

    def _create_pgm_input_transformers(self):  # pylint: disable-msg=too-many-locals
        """
        This function converts a Transformer Dataframe of PandaPower to a power-grid-model
        Transformer input array.

        Returns:
            a power-grid-model structured array for the Transformer component
        """
        pp_trafo = self.pp_input_data["trafo"]

        if pp_trafo.empty:
            return

        # Check for unsupported pandapower features
        if "tap_dependent_impedance" in pp_trafo.columns and any(pp_trafo["tap_dependent_impedance"]):
            raise RuntimeError("Tap dependent impedance is not supported in Power Grid Model")

        in_service = self._get_pp_attr("trafo", "in_service", True)
        parallel = self._get_pp_attr("trafo", "parallel", 1)
        sn_mva = self._get_pp_attr("trafo", "sn_mva")
        switch_states = self.get_switch_states("trafo")

        tap_side = self._get_pp_attr("trafo", "tap_side", None)
        tap_nom = self._get_pp_attr("trafo", "tap_neutral", np.nan)
        tap_pos = self._get_pp_attr("trafo", "tap_pos", np.nan)
        tap_size = self._get_tap_size(pp_trafo)
        # Do not use taps when mandatory tap data is not available
        no_taps = np.equal(tap_side, None) | np.isnan(tap_pos) | np.isnan(tap_nom) | np.isnan(tap_size)
        tap_nom[no_taps] = 0
        tap_pos[no_taps] = 0
        tap_size[no_taps] = 0
        tap_side[no_taps] = "hv"

        winding_types = self.get_trafo_winding_types()
        clocks = np.round(self._get_pp_attr("trafo", "shift_degree", 0.0) / 30) % 12
        # Default vector group for odd clocks = DYn and for even clocks = YNyn
        no_vector_groups = np.isnan(winding_types["winding_from"]) | np.isnan(winding_types["winding_to"])
        no_vector_groups_dyn = no_vector_groups & (clocks % 2)
        winding_types[no_vector_groups] = WindingType.wye_n
        winding_types["winding_from"][no_vector_groups_dyn] = WindingType.delta

        pgm_transformers = initialize_array(data_type="input", component_type="transformer", shape=len(pp_trafo))
        pgm_transformers["id"] = self._generate_ids("trafo", pp_trafo.index)
        pgm_transformers["from_node"] = self._get_pgm_ids("bus", self._get_pp_attr("trafo", "hv_bus"))
        pgm_transformers["from_status"] = in_service & switch_states["from"].values
        pgm_transformers["to_node"] = self._get_pgm_ids("bus", self._get_pp_attr("trafo", "lv_bus"))
        pgm_transformers["to_status"] = in_service & switch_states["to"].values
        pgm_transformers["u1"] = self._get_pp_attr("trafo", "vn_hv_kv") * 1e3
        pgm_transformers["u2"] = self._get_pp_attr("trafo", "vn_lv_kv") * 1e3
        pgm_transformers["sn"] = sn_mva * parallel * 1e6
        pgm_transformers["uk"] = self._get_pp_attr("trafo", "vk_percent") * 1e-2
        pgm_transformers["pk"] = self._get_pp_attr("trafo", "vkr_percent") * sn_mva * parallel * (1e6 * 1e-2)
        pgm_transformers["i0"] = self._get_pp_attr("trafo", "i0_percent") * 1e-2
        pgm_transformers["p0"] = self._get_pp_attr("trafo", "pfe_kw") * parallel * 1e3
        pgm_transformers["clock"] = clocks
        pgm_transformers["winding_from"] = winding_types["winding_from"]
        pgm_transformers["winding_to"] = winding_types["winding_to"]
        pgm_transformers["tap_nom"] = tap_nom
        pgm_transformers["tap_pos"] = tap_pos
        pgm_transformers["tap_side"] = self._get_transformer_tap_side(tap_side)
        pgm_transformers["tap_min"] = self._get_pp_attr("trafo", "tap_min", 0)
        pgm_transformers["tap_max"] = self._get_pp_attr("trafo", "tap_max", 0)
        pgm_transformers["tap_size"] = tap_size

        assert "transformer" not in self.pgm_input_data
        self.pgm_input_data["transformer"] = pgm_transformers

    def _create_pgm_input_three_winding_transformers(self):
        # pylint: disable=too-many-statements, disable-msg=too-many-locals
        """
        This function converts a Three Winding Transformer Dataframe of PandaPower to a power-grid-model
        Three Winding Transformer input array.

        Returns:
            a power-grid-model structured array for the Three Winding Transformer component
        """
        pp_trafo3w = self.pp_input_data["trafo3w"]

        if pp_trafo3w.empty:
            return

        # Check for unsupported pandapower features
        if "tap_dependent_impedance" in pp_trafo3w.columns and any(pp_trafo3w["tap_dependent_impedance"]):
            raise RuntimeError("Tap dependent impedance is not supported in Power Grid Model")  # pragma: no cover
        if "tap_at_star_point" in pp_trafo3w.columns and any(pp_trafo3w["tap_at_star_point"]):
            raise RuntimeError("Tap at star point is not supported in Power Grid Model")

        sn_hv_mva = self._get_pp_attr("trafo3w", "sn_hv_mva")
        sn_mv_mva = self._get_pp_attr("trafo3w", "sn_mv_mva")
        sn_lv_mva = self._get_pp_attr("trafo3w", "sn_lv_mva")
        in_service = self._get_pp_attr("trafo3w", "in_service", True)

        switch_states = self.get_trafo3w_switch_states(pp_trafo3w)

        tap_side = self._get_pp_attr("trafo3w", "tap_side", None)
        tap_nom = self._get_pp_attr("trafo3w", "tap_neutral", np.nan)
        tap_pos = self._get_pp_attr("trafo3w", "tap_pos", np.nan)
        tap_size = self._get_3wtransformer_tap_size(pp_trafo3w)
        # Do not use taps when mandatory tap data is not available
        no_taps = np.equal(tap_side, None) | np.isnan(tap_pos) | np.isnan(tap_nom) | np.isnan(tap_size)
        tap_nom[no_taps] = 0
        tap_pos[no_taps] = 0
        tap_size[no_taps] = 0
        tap_side[no_taps] = "hv"

        winding_types = self.get_trafo3w_winding_types()
        clocks_12 = np.round(self._get_pp_attr("trafo3w", "shift_mv_degree", 0.0) / 30.0) % 12
        clocks_13 = np.round(self._get_pp_attr("trafo3w", "shift_lv_degree", 0.0) / 30.0) % 12
        # Default vector group for odd clocks_12 = Yndx, for odd clocks_13 = Ynxd and for even clocks = YNxyn or YNynx
        no_vector_groups = (
            np.isnan(winding_types["winding_1"])
            | np.isnan(winding_types["winding_2"])
            | np.isnan(winding_types["winding_3"])
        )
        no_vector_groups_ynd2 = no_vector_groups & (clocks_12 % 2)
        no_vector_groups_ynd3 = no_vector_groups & (clocks_13 % 2)
        winding_types[no_vector_groups] = WindingType.wye_n
        winding_types["winding_2"][no_vector_groups_ynd2] = WindingType.delta
        winding_types["winding_3"][no_vector_groups_ynd3] = WindingType.delta

        pgm_3wtransformers = initialize_array(
            data_type="input", component_type="three_winding_transformer", shape=len(pp_trafo3w)
        )
        pgm_3wtransformers["id"] = self._generate_ids("trafo3w", pp_trafo3w.index)

        pgm_3wtransformers["node_1"] = self._get_pgm_ids("bus", self._get_pp_attr("trafo3w", "hv_bus"))
        pgm_3wtransformers["node_2"] = self._get_pgm_ids("bus", self._get_pp_attr("trafo3w", "mv_bus"))
        pgm_3wtransformers["node_3"] = self._get_pgm_ids("bus", self._get_pp_attr("trafo3w", "lv_bus"))
        pgm_3wtransformers["status_1"] = in_service & switch_states["side_1"].values
        pgm_3wtransformers["status_2"] = in_service & switch_states["side_2"].values
        pgm_3wtransformers["status_3"] = in_service & switch_states["side_3"].values
        pgm_3wtransformers["u1"] = self._get_pp_attr("trafo3w", "vn_hv_kv") * 1e3
        pgm_3wtransformers["u2"] = self._get_pp_attr("trafo3w", "vn_mv_kv") * 1e3
        pgm_3wtransformers["u3"] = self._get_pp_attr("trafo3w", "vn_lv_kv") * 1e3
        pgm_3wtransformers["sn_1"] = sn_hv_mva * 1e6
        pgm_3wtransformers["sn_2"] = sn_mv_mva * 1e6
        pgm_3wtransformers["sn_3"] = sn_lv_mva * 1e6
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
        pgm_3wtransformers["clock_12"] = clocks_12
        pgm_3wtransformers["clock_13"] = clocks_13
        pgm_3wtransformers["winding_1"] = winding_types["winding_1"]
        pgm_3wtransformers["winding_2"] = winding_types["winding_2"]
        pgm_3wtransformers["winding_3"] = winding_types["winding_3"]
        pgm_3wtransformers["tap_nom"] = tap_nom
        pgm_3wtransformers["tap_pos"] = tap_pos
        pgm_3wtransformers["tap_side"] = self._get_3wtransformer_tap_side(tap_side)
        pgm_3wtransformers["tap_min"] = self._get_pp_attr("trafo3w", "tap_min", 0)
        pgm_3wtransformers["tap_max"] = self._get_pp_attr("trafo3w", "tap_max", 0)
        pgm_3wtransformers["tap_size"] = tap_size

        assert "three_winding_transformer" not in self.pgm_input_data
        self.pgm_input_data["three_winding_transformer"] = pgm_3wtransformers

    def _create_pgm_input_links(self):
        """
        This function takes a Switch Dataframe of PandaPower, extracts the Switches which have Bus to Bus
        connection and converts them to a power-grid-model Link input array.

        Returns:
            a power-grid-model structured array for the Link component
        """
        pp_switches = self.pp_input_data["switch"]

        if pp_switches.empty:
            return

        # This should take all the switches which are b2b
        pp_switches = pp_switches[pp_switches["et"] == "b"]

        pgm_links = initialize_array(data_type="input", component_type="link", shape=len(pp_switches))
        pgm_links["id"] = self._generate_ids("switch", pp_switches.index, name="b2b_switches")
        pgm_links["from_node"] = self._get_pgm_ids("bus", pp_switches["bus"])
        pgm_links["to_node"] = self._get_pgm_ids("bus", pp_switches["element"])
        pgm_links["from_status"] = pp_switches["closed"]
        pgm_links["to_status"] = pp_switches["closed"]

        assert "link" not in self.pgm_input_data
        self.pgm_input_data["link"] = pgm_links

    def _create_pgm_input_storages(self):
        # TODO: create unit tests for the function
        pp_storage = self.pp_input_data["storage"]

        if pp_storage.empty:
            return

        raise NotImplementedError("Storage is not implemented yet!")

    def _create_pgm_input_impedances(self):
        # TODO: create unit tests for the function
        pp_impedance = self.pp_input_data["impedance"]

        if pp_impedance.empty:
            return

        raise NotImplementedError("Impedance is not implemented yet!")

    def _create_pgm_input_wards(self):
        # TODO: create unit tests for the function
        pp_wards = self.pp_input_data["ward"]

        if pp_wards.empty:
            return

        n_wards = len(pp_wards)
        in_service = self._get_pp_attr("ward", "in_service", True)
        bus = self._get_pp_attr("ward", "bus")

        pgm_sym_loads_from_ward = initialize_array(data_type="input", component_type="sym_load", shape=n_wards * 2)
        pgm_sym_loads_from_ward["id"][:n_wards] = self._generate_ids(
            "ward", pp_wards.index, name="ward_const_power_load"
        )
        pgm_sym_loads_from_ward["node"][:n_wards] = self._get_pgm_ids("bus", bus)
        pgm_sym_loads_from_ward["status"][:n_wards] = in_service
        pgm_sym_loads_from_ward["type"][:n_wards] = LoadGenType.const_power
        pgm_sym_loads_from_ward["p_specified"][:n_wards] = self._get_pp_attr("ward", "ps_mw") * 1e6
        pgm_sym_loads_from_ward["q_specified"][:n_wards] = self._get_pp_attr("ward", "qs_mvar") * 1e6

        pgm_sym_loads_from_ward["id"][-n_wards:] = self._generate_ids(
            "ward", pp_wards.index, name="ward_const_impedance_load"
        )
        pgm_sym_loads_from_ward["node"][-n_wards:] = self._get_pgm_ids("bus", bus)
        pgm_sym_loads_from_ward["status"][-n_wards:] = in_service
        pgm_sym_loads_from_ward["type"][-n_wards:] = LoadGenType.const_impedance
        pgm_sym_loads_from_ward["p_specified"][-n_wards:] = self._get_pp_attr("ward", "pz_mw") * 1e6
        pgm_sym_loads_from_ward["q_specified"][-n_wards:] = self._get_pp_attr("ward", "qz_mvar") * 1e6

        #  If input data of loads has already been filled then extend it with data of wards. If it is empty and there
        #  is no data about loads,then assign ward data to it
        if "sym_load" in self.pgm_input_data:
            symload_dtype = self.pgm_input_data["sym_load"].dtype
            self.pgm_input_data["sym_load"] = np.concatenate(  # pylint: disable=unexpected-keyword-arg
                [self.pgm_input_data["sym_load"], pgm_sym_loads_from_ward], dtype=symload_dtype
            )
        else:
            self.pgm_input_data["sym_load"] = pgm_sym_loads_from_ward

    def _create_pgm_input_xwards(self):
        # TODO: create unit tests for the function
        pp_xwards = self.pp_input_data["xward"]

        if pp_xwards.empty:
            return

        raise NotImplementedError("Extended Ward is not implemented yet!")

    def _create_pgm_input_motors(self):
        # TODO: create unit tests for the function
        pp_motors = self.pp_input_data["motor"]

        if pp_motors.empty:
            return

        pgm_sym_loads_from_motor = initialize_array(data_type="input", component_type="sym_load", shape=len(pp_motors))
        pgm_sym_loads_from_motor["id"] = self._generate_ids("motor", pp_motors.index, name="motor_load")
        pgm_sym_loads_from_motor["node"] = self._get_pgm_ids("bus", self._get_pp_attr("motor", "bus"))
        pgm_sym_loads_from_motor["status"] = self._get_pp_attr("motor", "in_service")
        pgm_sym_loads_from_motor["type"] = LoadGenType.const_power
        #  The formula for p_specified is pn_mech_mw /(efficiency_percent/100) * (loading_percent/100) * scaling * 1e6
        pgm_sym_loads_from_motor["p_specified"] = (
            self._get_pp_attr("motor", "pn_mech_mw")
            / self._get_pp_attr("motor", "efficiency_percent")
            * self._get_pp_attr("motor", "loading_percent")
            * self._get_pp_attr("motor", "scaling")
            * 1e6
        )
        p_spec = pgm_sym_loads_from_motor["p_specified"]
        pgm_sym_loads_from_motor["q_specified"] = np.sqrt(
            np.power(p_spec / self._get_pp_attr("motor", "cos_phi"), 2) - p_spec**2
        )

        #  If input data of loads has already been filled then extend it with data of motors. If it is empty and there
        #  is no data about loads,then assign motor data to it
        if "sym_load" in self.pgm_input_data:
            symload_dtype = self.pgm_input_data["sym_load"].dtype
            self.pgm_input_data["sym_load"] = np.concatenate(  # pylint: disable=unexpected-keyword-arg
                [self.pgm_input_data["sym_load"], pgm_sym_loads_from_motor], dtype=symload_dtype
            )
        else:
            self.pgm_input_data["sym_load"] = pgm_sym_loads_from_motor

    def _create_pgm_input_dclines(self):
        # TODO: create unit tests for the function
        pp_dcline = self.pp_input_data["dcline"]

        if pp_dcline.empty:
            return

        raise NotImplementedError("DC line is not implemented yet. power-grid-model does not support PV buses yet")

    def _create_pgm_input_generators(self):
        # TODO: create unit tests for the function
        pp_gen = self.pp_input_data["gen"]

        if pp_gen.empty:
            return

        raise NotImplementedError("Generators is not implemented yet. power-grid-model does not support PV buses yet")

    def _pp_buses_output(self):
        """
        This function converts a power-grid-model Node output array to a Bus Dataframe of PandaPower.

        Returns:
            a PandaPower Dataframe for the Bus component
        """
        # TODO: create unit tests for the function
        assert "res_bus" not in self.pp_output_data

        if "node" not in self.pgm_output_data or self.pgm_output_data["node"].size == 0:
            return

        pgm_nodes = self.pgm_output_data["node"]

        pp_output_buses = pd.DataFrame(
            columns=["vm_pu", "va_degree", "p_mw", "q_mvar"],
            index=self._get_pp_ids("bus", pgm_nodes["id"]),
        )

        pp_output_buses["vm_pu"] = pgm_nodes["u_pu"]
        pp_output_buses["va_degree"] = pgm_nodes["u_angle"] * (180.0 / np.pi)

        # p_to, p_from, q_to and q_from connected to the bus have to be summed up
        self._pp_buses_output__accumulate_power(pp_output_buses)

        self.pp_output_data["res_bus"] = pp_output_buses

    def _pp_buses_output__accumulate_power(self, pp_output_buses: pd.DataFrame):
        # TODO: create unit tests for the function
        """
        For each node, we accumulate the power for all connected branches and branch3s

        Args:
            pp_output_buses: a Pandapower output dataframe of Bus component

        Returns:
            accumulated power for each bus
        """

        # Let's define all the components and sides where nodes can be connected
        component_sides = {
            "line": [("from_node", "p_from", "q_from"), ("to_node", "p_to", "q_to")],
            "link": [("from_node", "p_from", "q_from"), ("to_node", "p_to", "q_to")],
            "transformer": [("from_node", "p_from", "q_from"), ("to_node", "p_to", "q_to")],
            "three_winding_transformer": [("node_1", "p_1", "q_1"), ("node_2", "p_2", "q_2"), ("node_3", "p_3", "q_3")],
        }

        # Set the initial powers to zero
        pp_output_buses["p_mw"] = 0.0
        pp_output_buses["q_mvar"] = 0.0

        # Now loop over all components, skipping the components that don't exist or don't contain data
        for component, sides in component_sides.items():
            if component not in self.pgm_output_data or self.pgm_output_data[component].size == 0:
                continue

            if component not in self.pgm_input_data:
                raise KeyError(f"PGM input_data is needed to accumulate output for {component}s.")

            for node_col, p_col, q_col in sides:
                # Select the columns that we are going to use
                component_data = pd.DataFrame(
                    zip(
                        self.pgm_input_data[component][node_col],
                        self.pgm_output_data[component][p_col],
                        self.pgm_output_data[component][q_col],
                    ),
                    columns=[node_col, p_col, q_col],
                )

                # Accumulate the powers and index by panda power bus index
                accumulated_data = component_data.groupby(node_col).sum()
                accumulated_data.index = self._get_pp_ids("bus", pd.Series(accumulated_data.index))

                # We might not have power data for each pp bus, so select only the indexes for which data is available
                idx = pp_output_buses.index.intersection(accumulated_data.index)

                # Now add the active and reactive powers to the pp busses
                # Note that the units are incorrect; for efficiency, unit conversions will be applied at the end.
                pp_output_buses.loc[idx, "p_mw"] -= accumulated_data[p_col]
                pp_output_buses.loc[idx, "q_mvar"] -= accumulated_data[q_col]

        # Finally apply the unit conversion (W -> MW and VAR -> MVAR)
        pp_output_buses["p_mw"] /= 1e6
        pp_output_buses["q_mvar"] /= 1e6

    def _pp_lines_output(self):
        """
        This function converts a power-grid-model Line output array to a Line Dataframe of PandaPower.

        Returns:
            a PandaPower Dataframe for the Line component
        """
        # TODO: create unit tests for the function
        assert "res_line" not in self.pp_output_data

        if "line" not in self.pgm_output_data or self.pgm_output_data["line"].size == 0:
            return

        pgm_input_lines = self.pgm_input_data["line"]
        pgm_output_lines = self.pgm_output_data["line"]

        if not np.array_equal(pgm_input_lines["id"], pgm_output_lines["id"]):
            raise ValueError("The output line ids should correspond to the input line ids")

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
            ],
            index=self._get_pp_ids("line", pgm_output_lines["id"]),
        )

        from_nodes = self.pgm_nodes_lookup.loc[pgm_input_lines["from_node"]]
        to_nodes = self.pgm_nodes_lookup.loc[pgm_input_lines["to_node"]]

        pp_output_lines["p_from_mw"] = pgm_output_lines["p_from"] * 1e-6
        pp_output_lines["q_from_mvar"] = pgm_output_lines["q_from"] * 1e-6
        pp_output_lines["p_to_mw"] = pgm_output_lines["p_to"] * 1e-6
        pp_output_lines["q_to_mvar"] = pgm_output_lines["q_to"] * 1e-6
        pp_output_lines["pl_mw"] = (pgm_output_lines["p_from"] + pgm_output_lines["p_to"]) * 1e-6
        pp_output_lines["ql_mvar"] = (pgm_output_lines["q_from"] + pgm_output_lines["q_to"]) * 1e-6
        pp_output_lines["i_from_ka"] = pgm_output_lines["i_from"] * 1e-3
        pp_output_lines["i_to_ka"] = pgm_output_lines["i_to"] * 1e-3
        pp_output_lines["i_ka"] = np.maximum(pgm_output_lines["i_from"], pgm_output_lines["i_to"]) * 1e-3
        pp_output_lines["vm_from_pu"] = from_nodes["u_pu"].values
        pp_output_lines["vm_to_pu"] = to_nodes["u_pu"].values
        pp_output_lines["va_from_degree"] = from_nodes["u_degree"].values
        pp_output_lines["va_to_degree"] = to_nodes["u_degree"].values
        pp_output_lines["loading_percent"] = pgm_output_lines["loading"] * 1e2

        self.pp_output_data["res_line"] = pp_output_lines

    def _pp_ext_grids_output(self):
        """
        This function converts a power-grid-model Source output array to an External Grid Dataframe of PandaPower.

        Returns:
            a PandaPower Dataframe for the External Grid component
        """
        assert "res_ext_grid" not in self.pp_output_data

        if "source" not in self.pgm_output_data or self.pgm_output_data["source"].size == 0:
            return

        pgm_output_sources = self.pgm_output_data["source"]

        pp_output_ext_grids = pd.DataFrame(
            columns=["p_mw", "q_mvar"], index=self._get_pp_ids("ext_grid", pgm_output_sources["id"])
        )
        pp_output_ext_grids["p_mw"] = pgm_output_sources["p"] * 1e-6
        pp_output_ext_grids["q_mvar"] = pgm_output_sources["q"] * 1e-6

        self.pp_output_data["res_ext_grid"] = pp_output_ext_grids

    def _pp_shunts_output(self):
        """
        This function converts a power-grid-model Shunt output array to a Shunt Dataframe of PandaPower.

        Returns:
            a PandaPower Dataframe for the Shunt component
        """
        # TODO: create unit tests for the function
        assert "res_shunt" not in self.pp_output_data

        if "shunt" not in self.pgm_output_data or self.pgm_output_data["shunt"].size == 0:
            return

        pgm_input_shunts = self.pgm_input_data["shunt"]

        pgm_output_shunts = self.pgm_output_data["shunt"]

        at_nodes = self.pgm_nodes_lookup.loc[pgm_input_shunts["node"]]

        pp_output_shunts = pd.DataFrame(
            columns=["p_mw", "q_mvar", "vm_pu"], index=self._get_pp_ids("shunt", pgm_output_shunts["id"])
        )
        pp_output_shunts["p_mw"] = pgm_output_shunts["p"] * 1e-6
        pp_output_shunts["q_mvar"] = pgm_output_shunts["q"] * 1e-6
        pp_output_shunts["vm_pu"] = at_nodes["u_pu"].values

        self.pp_output_data["res_shunt"] = pp_output_shunts

    def _pp_sgens_output(self):
        """
        This function converts a power-grid-model Symmetrical Generator output array to a Static Generator Dataframe of
        PandaPower.

        Returns:
            a PandaPower Dataframe for the Static Generator component
        """
        # TODO: create unit tests for the function
        assert "res_sgen" not in self.pp_output_data

        if "sym_gen" not in self.pgm_output_data or self.pgm_output_data["sym_gen"].size == 0:
            return

        pgm_output_sym_gens = self.pgm_output_data["sym_gen"]

        pp_output_sgens = pd.DataFrame(
            columns=["p_mw", "q_mvar"], index=self._get_pp_ids("sgen", pgm_output_sym_gens["id"])
        )
        pp_output_sgens["p_mw"] = pgm_output_sym_gens["p"] * 1e-6
        pp_output_sgens["q_mvar"] = pgm_output_sym_gens["q"] * 1e-6

        self.pp_output_data["res_sgen"] = pp_output_sgens

    def _pp_trafos_output(self):
        """
        This function converts a power-grid-model Transformer output array to a Transformer Dataframe of
        PandaPower.

        Returns:
            a PandaPower Dataframe for the Transformer component
        """
        # TODO: create unit tests for the function
        assert "res_trafo" not in self.pp_output_data

        if ("transformer" not in self.pgm_output_data or self.pgm_output_data["transformer"].size == 0) or (
            "trafo" not in self.pp_input_data or len(self.pp_input_data["trafo"]) == 0
        ):
            return

        pgm_input_transformers = self.pgm_input_data["transformer"]
        pp_input_transformers = self.pp_input_data["trafo"]
        pgm_output_transformers = self.pgm_output_data["transformer"]

        from_nodes = self.pgm_nodes_lookup.loc[pgm_input_transformers["from_node"]]
        to_nodes = self.pgm_nodes_lookup.loc[pgm_input_transformers["to_node"]]

        # Only derating factor used here. Sn is already being multiplied by parallel
        loading_multiplier = pp_input_transformers["df"]
        if self.trafo_loading == "current":
            ui_from = pgm_output_transformers["i_from"] * pgm_input_transformers["u1"]
            ui_to = pgm_output_transformers["i_to"] * pgm_input_transformers["u2"]
            loading = np.maximum(ui_from, ui_to) / pgm_input_transformers["sn"] * loading_multiplier * 1e2
        elif self.trafo_loading == "power":
            loading = pgm_output_transformers["loading"] * loading_multiplier * 1e2
        else:
            raise ValueError(f"Invalid transformer loading type: {str(self.trafo_loading)}")

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
                "va_hv_degree",
                "vm_lv_pu",
                "va_lv_degree",
                "loading_percent",
            ],
            index=self._get_pp_ids("trafo", pgm_output_transformers["id"]),
        )
        pp_output_trafos["p_hv_mw"] = pgm_output_transformers["p_from"] * 1e-6
        pp_output_trafos["q_hv_mvar"] = pgm_output_transformers["q_from"] * 1e-6
        pp_output_trafos["p_lv_mw"] = pgm_output_transformers["p_to"] * 1e-6
        pp_output_trafos["q_lv_mvar"] = pgm_output_transformers["q_to"] * 1e-6
        pp_output_trafos["pl_mw"] = (pgm_output_transformers["p_from"] + pgm_output_transformers["p_to"]) * 1e-6
        pp_output_trafos["ql_mvar"] = (pgm_output_transformers["q_from"] + pgm_output_transformers["q_to"]) * 1e-6
        pp_output_trafos["i_hv_ka"] = pgm_output_transformers["i_from"] * 1e-3
        pp_output_trafos["i_lv_ka"] = pgm_output_transformers["i_to"] * 1e-3
        pp_output_trafos["vm_hv_pu"] = from_nodes["u_pu"].values
        pp_output_trafos["vm_lv_pu"] = to_nodes["u_pu"].values
        pp_output_trafos["va_hv_degree"] = from_nodes["u_degree"].values
        pp_output_trafos["va_lv_degree"] = to_nodes["u_degree"].values
        pp_output_trafos["loading_percent"] = loading

        self.pp_output_data["res_trafo"] = pp_output_trafos

    def _pp_trafos3w_output(self):
        """
        This function converts a power-grid-model Three Winding Transformer output array to a Three Winding Transformer
        Dataframe of PandaPower.

        Returns:
            a PandaPower Dataframe for the Three Winding Transformer component
        """
        # TODO: create unit tests for the function
        assert "res_trafo3w" not in self.pp_output_data

        if (
            "three_winding_transformer" not in self.pgm_output_data
            or self.pgm_output_data["three_winding_transformer"].size == 0
        ):
            return

        pgm_input_transformers3w = self.pgm_input_data["three_winding_transformer"]

        pgm_output_transformers3w = self.pgm_output_data["three_winding_transformer"]

        nodes_1 = self.pgm_nodes_lookup.loc[pgm_input_transformers3w["node_1"]]
        nodes_2 = self.pgm_nodes_lookup.loc[pgm_input_transformers3w["node_2"]]
        nodes_3 = self.pgm_nodes_lookup.loc[pgm_input_transformers3w["node_3"]]

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
            index=self._get_pp_ids("trafo3w", pgm_output_transformers3w["id"]),
        )

        pp_output_trafos3w["p_hv_mw"] = pgm_output_transformers3w["p_1"] * 1e-6
        pp_output_trafos3w["q_hv_mvar"] = pgm_output_transformers3w["q_1"] * 1e-6
        pp_output_trafos3w["p_mv_mw"] = pgm_output_transformers3w["p_2"] * 1e-6
        pp_output_trafos3w["q_mv_mvar"] = pgm_output_transformers3w["q_2"] * 1e-6
        pp_output_trafos3w["p_lv_mw"] = pgm_output_transformers3w["p_3"] * 1e-6
        pp_output_trafos3w["q_lv_mvar"] = pgm_output_transformers3w["q_3"] * 1e-6
        pp_output_trafos3w["pl_mw"] = (
            pgm_output_transformers3w["p_1"] + pgm_output_transformers3w["p_2"] + pgm_output_transformers3w["p_3"]
        ) * 1e-6
        pp_output_trafos3w["ql_mvar"] = (
            pgm_output_transformers3w["q_1"] + pgm_output_transformers3w["q_2"] + pgm_output_transformers3w["q_3"]
        ) * 1e-6
        pp_output_trafos3w["i_hv_ka"] = pgm_output_transformers3w["i_1"] * 1e-3
        pp_output_trafos3w["i_mv_ka"] = pgm_output_transformers3w["i_2"] * 1e-3
        pp_output_trafos3w["i_lv_ka"] = pgm_output_transformers3w["i_3"] * 1e-3
        pp_output_trafos3w["vm_hv_pu"] = nodes_1["u_pu"].values
        pp_output_trafos3w["vm_mv_pu"] = nodes_2["u_pu"].values
        pp_output_trafos3w["vm_lv_pu"] = nodes_3["u_pu"].values
        pp_output_trafos3w["va_hv_degree"] = nodes_1["u_degree"].values
        pp_output_trafos3w["va_mv_degree"] = nodes_2["u_degree"].values
        pp_output_trafos3w["va_lv_degree"] = nodes_3["u_degree"].values
        pp_output_trafos3w["loading_percent"] = pgm_output_transformers3w["loading"] * 1e2

        self.pp_output_data["res_trafo3w"] = pp_output_trafos3w

    def _pp_asym_loads_output(self):
        """
        This function converts a power-grid-model Asymmetrical Load output array to an Asymmetrical Load Dataframe of
        PandaPower.

        Returns:
            a PandaPower Dataframe for the Asymmetrical Load component
        """
        # TODO: create unit tests for the function
        assert "res_asymmetric_load" not in self.pp_output_data

        if "asym_load" not in self.pgm_output_data or self.pgm_output_data["asym_load"].size == 0:
            return

        pgm_output_asym_loads = self.pgm_output_data["asym_load"]

        pp_asym_output_loads = pd.DataFrame(
            columns=["p_mw", "q_mvar"],
            index=self._get_pp_ids("asymmetric_load", pgm_output_asym_loads["id"]),
        )

        pp_asym_output_loads["p_mw"] = pgm_output_asym_loads["p"] * 1e-6
        pp_asym_output_loads["q_mvar"] = pgm_output_asym_loads["q"] * 1e-6

        self.pp_output_data["res_asymmetric_load"] = pp_asym_output_loads

    def _pp_asym_gens_output(self):
        """
        This function converts a power-grid-model Asymmetrical Generator output array to an Asymmetric Static Generator
        Dataframe of PandaPower.

        Returns:
            a PandaPower Dataframe for the Asymmetric Static Generator component
        """
        # TODO: create unit tests for the function
        assert "res_asymmetric_sgen" not in self.pp_output_data

        if "asym_gen" not in self.pgm_output_data or self.pgm_output_data["asym_gen"].size == 0:
            return

        pgm_output_asym_gens = self.pgm_output_data["asym_gen"]

        pp_output_asym_gens = pd.DataFrame(
            columns=["p_mw", "q_mvar"],
            index=self._get_pp_ids("asymmetric_sgen", pgm_output_asym_gens["id"]),
        )

        pp_output_asym_gens["p_mw"] = pgm_output_asym_gens["p"] * 1e-6
        pp_output_asym_gens["q_mvar"] = pgm_output_asym_gens["q"] * 1e-6

        self.pp_output_data["res_asymmetric_sgen"] = pp_output_asym_gens

    def _pp_loads_output(self):
        """
        This function converts a power-grid-model Symmetrical Load output array to a Load Dataframe of PandaPower.

        Returns:
            a PandaPower Dataframe for the Load component
        """
        load_id_names = ["const_power", "const_impedance", "const_current"]
        assert "res_load" not in self.pp_output_data

        if (
            "sym_load" not in self.pgm_output_data
            or self.pgm_output_data["sym_load"].size == 0
            or ("load", load_id_names[0]) not in self.idx
        ):
            return

        self._pp_load_result_accumulate(
            pp_component_name="load", load_id_names=["const_power", "const_impedance", "const_current"]
        )

    def _pp_ward_output(self):
        load_id_names = ["ward_const_power_load", "ward_const_impedance_load"]
        assert "res_ward" not in self.pp_output_data

        if (
            "sym_load" not in self.pgm_output_data
            or self.pgm_output_data["sym_load"].size == 0
            or ("ward", load_id_names[0]) not in self.idx
        ):
            return

        self._pp_load_result_accumulate(pp_component_name="ward", load_id_names=load_id_names)
        # TODO Find a better way for mapping vm_pu from bus
        # self.pp_output_data["res_ward"]["vm_pu"] = np.nan

    def _pp_motor_output(self):
        load_id_names = ["motor_load"]

        assert "res_motor" not in self.pp_output_data

        if (
            "sym_load" not in self.pgm_output_data
            or self.pgm_output_data["sym_load"].size == 0
            or ("motor", load_id_names[0]) not in self.idx
        ):
            return

        self._pp_load_result_accumulate(pp_component_name="motor", load_id_names=load_id_names)

    def _pp_load_result_accumulate(self, pp_component_name: str, load_id_names: List[str]):
        """
        This function converts a power-grid-model Symmetrical Load output array to a respective Dataframe of PandaPower.

        Returns:
            a PandaPower Dataframe for the Load component
        """
        # Create a DataFrame wih all the pgm output loads and index it in the pgm id
        pgm_output_loads = self.pgm_output_data["sym_load"]
        all_loads = pd.DataFrame(pgm_output_loads, index=pgm_output_loads["id"])

        # Create an empty DataFrame with two columns p and q to accumulate all the loads per pp id
        accumulated_loads = pd.DataFrame(columns=["p", "q"], dtype=np.float64)

        # Loop over the load types;
        #   find the pgm ids
        #   select those loads
        #   replace the index by the pp ids
        #   add the p and q columns to the accumulator DataFrame
        for load_type in load_id_names:
            pgm_load_ids = self._get_pgm_ids(pp_component_name, name=load_type)
            selected_loads = all_loads.loc[pgm_load_ids]
            selected_loads.index = pgm_load_ids.index  # The index contains the pp ids
            accumulated_loads = accumulated_loads.add(selected_loads[["p", "q"]], fill_value=0.0)

        # Multiply the values and rename the columns to match pandapower
        accumulated_loads *= 1e-6
        accumulated_loads.columns = ["p_mw", "q_mvar"]

        # Store the results, while assuring that we are not overwriting any data
        assert pp_component_name not in self.pp_output_data
        self.pp_output_data["res_" + pp_component_name] = accumulated_loads

    def _pp_switches_output(self):
        """
        This function converts a power-grid-model links, lines, transformers, transformers3w output array
        to res_switch Dataframe of PandaPower.
        Switch results are only possible at round conversions. ie, input switch data is available
        """
        switch_data_unavailable = "switch" not in self.pp_input_data
        links_absent = "link" not in self.pgm_output_data or self.pgm_output_data["link"].size == 0
        rest_switches_absent = {
            pp_comp: ("res_" + pp_comp not in self.pp_output_data) for pp_comp in ["line", "trafo", "trafo3w"]
        }
        if (all(rest_switches_absent.values()) and links_absent) or switch_data_unavailable:
            return

        def join_currents(table: str, bus_name: str, i_name: str) -> pd.DataFrame:
            # Create a dataframe of element: input table index, bus: input branch bus, current: output current
            single_df = self.pp_input_data[table][[bus_name]]
            single_df = single_df.join(self.pp_output_data["res_" + table][i_name])
            single_df.columns = ["bus", "i_ka"]
            single_df["element"] = single_df.index
            single_df["et"] = table_to_et[table]
            return single_df

        switch_attrs = {
            "trafo": {"hv_bus": "i_hv_ka", "lv_bus": "i_lv_ka"},
            "trafo3w": {"hv_bus": "i_hv_ka", "mv_bus": "i_mv_ka", "lv_bus": "i_lv_ka"},
            "line": {"from_bus": "i_from_ka", "to_bus": "i_to_ka"},
        }
        table_to_et = {"trafo": "t", "trafo3w": "t3", "line": "l"}

        # Prepare output dataframe, save index for later
        pp_switches_output = self.pp_input_data["switch"]
        pp_switches_output_index = pp_switches_output.index

        # Combine all branch bus, current and et in one dataframe
        all_i_df = pd.concat(
            [
                join_currents(table, bus_name, i_name)
                if not rest_switches_absent[table]
                else pd.DataFrame(columns=["bus", "element", "et", "i_ka"])
                for table, attr_names in switch_attrs.items()
                for bus_name, i_name in attr_names.items()
            ]
        )
        # Merge on input data to get current and drop other columns
        pp_switches_output = pd.merge(
            pp_switches_output,
            all_i_df,
            how="left",
            left_on=["bus", "element", "et"],
            right_on=["bus", "element", "et"],
        )
        pp_switches_output = pp_switches_output[["i_ka"]]
        pp_switches_output.set_index(pp_switches_output_index, inplace=True)
        pp_switches_output["loading_percent"] = np.nan

        # For et=b, ie bus to bus switches, links are created. get result from them
        if not links_absent:
            links = self.pgm_output_data["link"]
            # For links, i_from = i_to = i_ka / 1e3
            link_ids = self._get_pp_ids("switch", links["id"], "b2b_switches")
            pp_switches_output["i_ka"][link_ids] = links["i_from"] * 1e-3

        assert "res_switch" not in self.pp_output_data
        self.pp_output_data["res_switch"] = pp_switches_output

    def _generate_ids(self, pp_table: str, pp_idx: pd.Index, name: Optional[str] = None) -> np.arange:
        """
        Generate numerical power-grid-model IDs for a PandaPower component

        Args:
            pp_table: Table name (e.g. "bus")
            pp_idx: PandaPower component identifier

        Returns:
            the generated IDs
        """
        key = (pp_table, name)
        assert key not in self.idx_lookup
        n_objects = len(pp_idx)
        pgm_idx = np.arange(start=self.next_idx, stop=self.next_idx + n_objects, dtype=np.int32)
        self.idx[key] = pd.Series(pgm_idx, index=pp_idx)
        self.idx_lookup[key] = pd.Series(pp_idx, index=pgm_idx)
        self.next_idx += n_objects
        return pgm_idx

    def _get_pgm_ids(
        self, pp_table: str, pp_idx: Optional[Union[pd.Series, np.array]] = None, name: Optional[str] = None
    ) -> pd.Series:
        """
        Get numerical power-grid-model IDs for a PandaPower component

        Args:
            pp_table: Table name (e.g. "bus")
            pp_idx: PandaPower component identifier

        Returns:
            the power-grid-model IDs if they were previously generated
        """
        key = (pp_table, name)
        if key not in self.idx:
            raise KeyError(f"No indexes have been created for '{pp_table}' (name={name})!")
        if pp_idx is None:
            return self.idx[key]
        return self.idx[key][pp_idx]

    def _get_pp_ids(self, pp_table: str, pgm_idx: Optional[pd.Series] = None, name: Optional[str] = None) -> pd.Series:
        """
        Get numerical PandaPower IDs for a PandaPower component

        Args:
            pp_table: Table name (e.g. "bus")
            pgm_idx: power-grid-model component identifier

        Returns:
            the PandaPower IDs if they were previously generated
        """
        key = (pp_table, name)
        if key not in self.idx_lookup:
            raise KeyError(f"No indexes have been created for '{pp_table}' (name={name})!")
        if pgm_idx is None:
            return self.idx_lookup[key]
        return self.idx_lookup[key][pgm_idx]

    @staticmethod
    def _get_tap_size(pp_trafo: pd.DataFrame) -> np.ndarray:
        """
        Calculate the "tap size" of Transformers

        Args:
            pp_trafo: PandaPower dataframe with information about the Transformers in
            the Network (e.g. "hv_bus", "i0_percent")

        Returns:
            the "tap size" of Transformers
        """
        tap_side_hv = np.array(pp_trafo["tap_side"] == "hv")
        tap_side_lv = np.array(pp_trafo["tap_side"] == "lv")
        tap_step_multiplier = pp_trafo["tap_step_percent"] * (1e-2 * 1e3)

        tap_size = np.empty(shape=len(pp_trafo), dtype=np.float64)
        tap_size[tap_side_hv] = tap_step_multiplier[tap_side_hv] * pp_trafo["vn_hv_kv"][tap_side_hv]
        tap_size[tap_side_lv] = tap_step_multiplier[tap_side_lv] * pp_trafo["vn_lv_kv"][tap_side_lv]

        return tap_size

    @staticmethod
    def _get_transformer_tap_side(tap_side: np.ndarray) -> np.ndarray:
        """
        Get the enumerated "tap side" of Transformers

        Args:
            tap_side: PandaPower series with information about the "tap_side" attribute

        Returns:
            the enumerated "tap side"
        """

        # Both "hv" and None should be converted to BranchSide.from_side
        new_tap_side = np.full(shape=tap_side.shape, fill_value=BranchSide.from_side)
        new_tap_side[tap_side == "lv"] = BranchSide.to_side

        return new_tap_side

    @staticmethod
    def _get_3wtransformer_tap_side(tap_side: np.ndarray) -> np.ndarray:
        """
        Get the enumerated "tap side" of Three Winding Transformers

        Args:
            tap_side: PandaPower series with information about the "tap_side" attribute

        Returns:
            the enumerated "tap side"
        """
        # Both "hv" and None should be converted to Branch3Side.side_1
        new_tap_side = np.full(shape=tap_side.shape, fill_value=Branch3Side.side_1)
        new_tap_side[tap_side == "mv"] = Branch3Side.side_2
        new_tap_side[tap_side == "lv"] = Branch3Side.side_3

        return new_tap_side

    @staticmethod
    def _get_3wtransformer_tap_size(pp_3wtrafo: pd.DataFrame) -> np.ndarray:
        """
        Calculate the "tap size" of Three Winding Transformers

        Args:
            pp_3wtrafo: PandaPower dataframe with information about the Three Winding Transformers in
            the Network (e.g. "hv_bus", "i0_percent")

        Returns:
            the "tap size" of Three Winding Transformers
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
            the "closed" value of a Switch
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
        return pd.Series(switch_state["closed"])

    def get_switch_states(self, pp_table: str) -> pd.DataFrame:
        """
        Return switch states of either Lines or Transformers

        Args:
            pp_table: Table name (e.g. "bus")

        Returns:
            the switch states of either Lines or Transformers
        """
        if pp_table == "line":
            element_type = "l"
            bus1 = "from_bus"
            bus2 = "to_bus"
        elif pp_table == "trafo":
            element_type = "t"
            bus1 = "hv_bus"
            bus2 = "lv_bus"
        else:
            raise KeyError(f"Can't get switch states for {pp_table}")

        component = self.pp_input_data[pp_table]
        component["index"] = component.index

        # Select the appropriate switches and columns
        pp_switches = self.pp_input_data["switch"]
        pp_switches = pp_switches[pp_switches["et"] == element_type]
        pp_switches = pp_switches[["element", "bus", "closed"]]

        pp_from_switches = self.get_individual_switch_states(component[["index", bus1]], pp_switches, bus1)
        pp_to_switches = self.get_individual_switch_states(component[["index", bus2]], pp_switches, bus2)

        return pd.DataFrame({"from": pp_from_switches, "to": pp_to_switches})

    def get_trafo3w_switch_states(self, trafo3w: pd.DataFrame) -> pd.DataFrame:
        """
        Return switch states of Three Winding Transformers

        Args:
            trafo3w: PandaPower dataframe with information about the Three Winding Transformers.

        Returns:
            the switch states of Three Winding Transformers
        """
        element_type = "t3"
        bus1 = "hv_bus"
        bus2 = "mv_bus"
        bus3 = "lv_bus"
        trafo3w["index"] = trafo3w.index

        # Select the appropriate switches and columns
        pp_switches = self.pp_input_data["switch"]
        pp_switches = pp_switches[pp_switches["et"] == element_type]
        pp_switches = pp_switches[["element", "bus", "closed"]]

        # Join the switches with the three winding trafo three times, for the hv_bus, mv_bus and once for the lv_bus
        pp_1_switches = self.get_individual_switch_states(trafo3w[["index", bus1]], pp_switches, bus1)
        pp_2_switches = self.get_individual_switch_states(trafo3w[["index", bus2]], pp_switches, bus2)
        pp_3_switches = self.get_individual_switch_states(trafo3w[["index", bus3]], pp_switches, bus3)

        return pd.DataFrame(
            data={"side_1": pp_1_switches, "side_2": pp_2_switches, "side_3": pp_3_switches}, index=trafo3w.index
        )

    def get_trafo_winding_types(self) -> pd.DataFrame:
        """
        This function extracts Transformers' "winding_type" attribute through "vector_group" attribut.

        Returns:
            the "from" and "to" winding types of a transformer
        """

        @lru_cache
        def vector_group_to_winding_types(vector_group: str) -> pd.Series:
            match = TRAFO_CONNECTION_RE.fullmatch(vector_group)
            if not match:
                raise ValueError(f"Invalid transformer connection string: '{vector_group}'")
            winding_from = get_winding(match.group(1)).value
            winding_to = get_winding(match.group(2)).value
            return pd.Series([winding_from, winding_to])

        trafo = self.pp_input_data["trafo"]
        col_names = ["winding_from", "winding_to"]
        if "vector_group" not in trafo:
            return pd.DataFrame(np.full(shape=(len(trafo), 2), fill_value=np.nan), columns=col_names)
        trafo = trafo["vector_group"].apply(vector_group_to_winding_types)
        trafo.columns = col_names
        return trafo

    def get_trafo3w_winding_types(self) -> pd.DataFrame:
        """
        This function extracts Three Winding Transformers' "winding_type" attribute through "vector_group" attribute.

        Returns:
            the three winding types of Three Winding Transformers
        """

        @lru_cache
        def vector_group_to_winding_types(vector_group: str) -> pd.Series:
            match = TRAFO3_CONNECTION_RE.fullmatch(vector_group)
            if not match:
                raise ValueError(f"Invalid transformer connection string: '{vector_group}'")
            winding_1 = get_winding(match.group(1)).value
            winding_2 = get_winding(match.group(2)).value
            winding_3 = get_winding(match.group(4)).value
            return pd.Series([winding_1, winding_2, winding_3])

        trafo3w = self.pp_input_data["trafo3w"]
        col_names = ["winding_1", "winding_2", "winding_3"]
        if "vector_group" not in trafo3w:
            return pd.DataFrame(np.full(shape=(len(trafo3w), 3), fill_value=np.nan), columns=col_names)
        trafo3w = trafo3w["vector_group"].apply(vector_group_to_winding_types)
        trafo3w.columns = col_names
        return trafo3w

    def _get_pp_attr(self, table: str, attribute: str, default: Optional[Union[float, bool, str]] = None) -> np.ndarray:
        """
        Returns the selected PandaPower attribute from the selected PandaPower table.

        Args:
            table: Table name (e.g. "bus")
            attribute: an attribute from the table (e.g "vn_kv")

        Returns:
            the selected PandaPower attribute from the selected PandaPower table
        """
        pp_component_data = self.pp_input_data[table]

        # If the attribute does not exists, return the default value
        # (assume that broadcasting is handled by the caller / numpy)
        if attribute not in pp_component_data:
            if default is None:
                raise KeyError(f"No '{attribute}' value for '{table}'.")
            return np.array([default])

        attr_data = pp_component_data[attribute]

        # If any of the attribute values are missing, and a default is supplied, fill the nans with the default value
        nan_values = np.equal(attr_data, None) if attr_data.dtype is np.dtype("O") else np.isnan(attr_data)
        if any(nan_values):
            attr_data = attr_data.fillna(value=default, inplace=False)

        return attr_data.to_numpy()

    def get_id(self, pp_table: str, pp_idx: int, name: Optional[str] = None) -> int:
        """
        Get a numerical ID previously associated with the supplied table / index combination

        Args:
            pp_table: Table name (e.g. "bus")
            pp_idx: PandaPower component identifier
            name: Optional component name (e.g. "internal_node")

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
