# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
# pylint: disable = too-many-lines
"""
Panda Power Converter
"""

import logging
from collections.abc import MutableMapping
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version

import numpy as np
import pandas as pd
import structlog
from packaging.version import Version
from power_grid_model import (
    AttributeType,
    Branch3Side,
    BranchSide,
    ComponentType,
    DatasetType,
    LoadGenType,
    WindingType,
    initialize_array,
    power_grid_meta_data,
)
from power_grid_model.data_types import Dataset, SingleDataset

from power_grid_model_io._enum import PandapowerAttribute as _PpAttr, PandapowerTable as _PpTable
from power_grid_model_io.converters.base_converter import BaseConverter
from power_grid_model_io.data_types import ExtraInfo
from power_grid_model_io.functions import get_winding
from power_grid_model_io.utils.parsing import is_node_ref, parse_trafo3_connection, parse_trafo_connection

PandaPowerData = MutableMapping[str, pd.DataFrame]

logger = structlog.get_logger(__file__)

ARRAY_2D = 2
PP_COMPATIBILITY_VERSION_3_2_0 = Version("3.2.0")
PP_COMPATIBILITY_VERSION_3_4_0 = Version("3.4.0")
try:
    PP_CONVERSION_VERSION = Version(version("pandapower"))
except PackageNotFoundError:
    PP_CONVERSION_VERSION = PP_COMPATIBILITY_VERSION_3_4_0  # assume latest compatible version by default


def get_loss_params_3ph():
    if PP_CONVERSION_VERSION < PP_COMPATIBILITY_VERSION_3_2_0:
        loss_params = [
            _PpAttr.p_a_l_mw,
            _PpAttr.q_a_l_mvar,
            _PpAttr.p_b_l_mw,
            _PpAttr.q_b_l_mvar,
            _PpAttr.p_c_l_mw,
            _PpAttr.q_c_l_mvar,
        ]
    else:
        loss_params = [
            _PpAttr.pl_a_mw,
            _PpAttr.ql_a_mvar,
            _PpAttr.pl_b_mw,
            _PpAttr.ql_b_mvar,
            _PpAttr.pl_c_mw,
            _PpAttr.ql_c_mvar,
        ]
    return loss_params


# pylint: disable=too-many-instance-attributes
class PandaPowerConverter(BaseConverter[PandaPowerData]):
    """
    Panda Power Converter
    """

    __slots__ = (
        "idx",
        "idx_lookup",
        "next_idx",
        "pgm_input_data",
        "pp_input_data",
        "system_frequency",
    )

    def __init__(
        self,
        system_frequency: float = 50.0,
        trafo_loading: str = "current",
        log_level: int = logging.INFO,
    ):
        """
        Prepare some member variables

        Args:
            system_frequency: fundamental frequency of the alternating current and voltage in the Network measured in Hz
        """
        super().__init__(source=None, destination=None, log_level=log_level)
        self.trafo_loading = trafo_loading
        self.system_frequency: float = system_frequency
        self.pp_input_data: PandaPowerData = {}
        self.pgm_input_data: SingleDataset = {}
        self.pp_output_data: PandaPowerData = {}
        self.pgm_output_data: SingleDataset = {}
        self.pgm_nodes_lookup: pd.DataFrame = pd.DataFrame()
        self.idx: dict[tuple[str, str | None], pd.Series] = {}
        self.idx_lookup: dict[tuple[str, str | None], pd.Series] = {}
        self.next_idx = 0

    def _parse_data(
        self,
        data: PandaPowerData,
        data_type: DatasetType,
        extra_info: ExtraInfo | None = None,
    ) -> Dataset:
        """
        Set up for conversion from PandaPower to power-grid-model

        Args:
            data: PandaPowerData, i.e. a dictionary with the components as keys and pd.DataFrames as values, with
            attribute names as columns and their values in the table
            data_type: power-grid-model data type, i.e. DatasetType.input or DatasetType.update
            extra_info: an optional dictionary where extra component info (that can't be specified in
            power-grid-model data) can be specified

        Returns:
            Converted power-grid-model data
        """

        # Clear pgm data
        self.pgm_input_data = {}
        self.idx_lookup = {}
        self.next_idx = 0

        # Set pandapower data
        self.pp_input_data = data

        # Convert
        if data_type == DatasetType.input:
            self._create_input_data()
        else:
            raise ValueError(f"Data type: '{data_type}' is not implemented")

        # Construct extra_info
        if extra_info is not None:
            self._fill_pgm_extra_info(extra_info=extra_info)
            self._fill_pp_extra_info(extra_info=extra_info)

        return self.pgm_input_data

    def _serialize_data(self, data: Dataset, extra_info: ExtraInfo | None) -> PandaPowerData:
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
        def pgm_output_dtype_checker(check_type: DatasetType | str) -> bool:
            return all(
                (
                    comp_array.dtype == power_grid_meta_data[DatasetType[check_type]][component]
                    for component, comp_array in self.pgm_output_data.items()
                )
            )

        # Convert
        if pgm_output_dtype_checker(DatasetType.sym_output):
            self._create_output_data()
        elif pgm_output_dtype_checker(DatasetType.asym_output):
            self._create_output_data_3ph()
        else:
            raise TypeError("Invalid output data dictionary supplied.")

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

    def _fill_pgm_extra_info(self, extra_info: ExtraInfo):
        """
        Fills in extra information of power-grid-model input after conversion from pandapower to the extra_info dict

        Args:
            extra_info: The extra info dict
        """
        for (pp_table, name), indices in self.idx_lookup.items():
            for pgm_id, pp_idx in zip(indices.index, indices):
                if name:
                    extra_info[pgm_id] = {
                        "id_reference": {
                            "table": pp_table,
                            "name": name,
                            "index": pp_idx,
                        }
                    }
                else:
                    extra_info[pgm_id] = {"id_reference": {"table": pp_table, "index": pp_idx}}

        extra_cols = [AttributeType.i_n]
        for component_data in self.pgm_input_data.values():
            for attr_name in component_data.dtype.names:
                if not is_node_ref(attr_name) and attr_name not in extra_cols:
                    continue
                for pgm_id, node_id in component_data[[AttributeType.id, attr_name]]:
                    if pgm_id not in extra_info:
                        extra_info[pgm_id] = {}
                    if "pgm_input" not in extra_info[pgm_id]:
                        extra_info[pgm_id]["pgm_input"] = {}
                    extra_info[pgm_id]["pgm_input"][attr_name] = node_id

    def _fill_pp_extra_info(self, extra_info: ExtraInfo):
        """
        Fills extra information from pandapower input dataframes not available in power-grid-model input
        to the extra_info dict.
        Currently, it is possible to only store the derating factor (df) of trafo.

        Args:
            extra_info: The extra info dict
        """
        pp_input = {_PpTable.trafo: {"df"}}
        for pp_table, pp_attr in pp_input.items():
            if (
                pp_table in self.pp_input_data
                and pp_attr & set(self.pp_input_data[pp_table].columns)
                and len(self.pp_input_data[pp_table]) > 0
            ):
                pgm_ids = self._get_pgm_ids(pp_table=pp_table)
                pp_extra_data = self.pp_input_data[pp_table][list(pp_attr)]
                pp_extra_data.index = pgm_ids
                for pgm_id, pp_element in pp_extra_data.iterrows():
                    if pgm_id not in extra_info:
                        extra_info[pgm_id] = {}
                    if "pp_input" not in extra_info[pgm_id]:
                        extra_info[pgm_id]["pp_input"] = {}
                    for attr in pp_attr:
                        extra_info[pgm_id]["pp_input"][attr] = pp_element[attr]

    def _extra_info_to_idx_lookup(self, extra_info: ExtraInfo):
        """
        Converts extra component info into idx_lookup

        Args:
            extra_info: a dictionary where the original panda power ids are stored
        """
        self.idx = {}
        self.idx_lookup = {}
        pgm_to_pp_id: dict[tuple[str, str | None], list[tuple[int, int]]] = {}
        for pgm_idx, extra in extra_info.items():
            if "id_reference" not in extra:
                continue
            if not isinstance(extra["id_reference"], dict):
                raise TypeError(
                    f"Expected 'id_reference' to be a dict for pgm_id {pgm_idx}, "
                    f"got {type(extra['id_reference']).__name__}"
                )
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

    def _extra_info_to_pgm_input_data(self, extra_info: ExtraInfo):  # pylint: disable-msg=too-many-locals
        """
        Converts extra component info into node_lookup

        Args:
            extra_info: a dictionary where the node reference ids are stored
        """
        if self.pgm_input_data:
            raise ValueError("pgm_input_data should be empty")
        if not self.pgm_output_data:
            raise ValueError("pgm_output_data should not be empty")

        dtype = np.int32
        other_cols_dtype = np.float64
        nan = np.iinfo(dtype).min
        all_other_cols = [AttributeType.i_n]
        for component, data in self.pgm_output_data.items():
            input_cols = power_grid_meta_data[DatasetType.input][component].dtype.names
            if input_cols is None:
                input_cols = tuple()

            node_cols = [col for col in input_cols if is_node_ref(col)]
            other_cols = [col for col in input_cols if col in all_other_cols]
            if not node_cols + other_cols:
                continue
            num_cols = 1 + len(node_cols)
            num_other_cols = len(other_cols)
            ref = np.full(
                shape=len(data),
                fill_value=nan,
                dtype={
                    "names": [AttributeType.id, *node_cols, *other_cols],
                    "formats": [dtype] * num_cols + [other_cols_dtype] * num_other_cols,
                },
            )
            for i, pgm_id in enumerate(data[AttributeType.id]):
                extra = extra_info[pgm_id].get("pgm_input", {})
                ref[i] = (pgm_id, *tuple(extra[col] for col in node_cols + other_cols))
            self.pgm_input_data[component] = ref

    def _extra_info_to_pp_input_data(self, extra_info: ExtraInfo):
        """
        Converts extra component info into node_lookup
        Currently, it is possible to only retrieve the derating factor (df) of trafo.

        Args:
            extra_info: a dictionary where the node reference ids are stored
        """
        if self.pp_input_data:
            raise ValueError("pp_input_data should be empty")
        if not self.pgm_output_data:
            raise ValueError("pgm_output_data should not be empty")

        if ComponentType.transformer not in self.pgm_output_data:
            return

        pgm_ids = self.pgm_output_data[ComponentType.transformer][AttributeType.id]
        pp_ids = self._get_pp_ids(pp_table=_PpTable.trafo, pgm_idx=pgm_ids)
        derating_factor = (extra_info.get(pgm_id, {}).get("pp_input", {}).get("df", np.nan) for pgm_id in pgm_ids)
        self.pp_input_data = {_PpTable.trafo: pd.DataFrame(derating_factor, columns=["df"], index=pp_ids)}

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
                "u_pu": self.pgm_output_data[ComponentType.node][AttributeType.u_pu],
                "u_degree": self.pgm_output_data[ComponentType.node][AttributeType.u_angle] * (180.0 / np.pi),
            },
            index=self.pgm_output_data[ComponentType.node][AttributeType.id],
        )

        self._pp_buses_output()
        self._pp_lines_output()
        self._pp_ext_grids_output()
        self._pp_load_elements_output(element=_PpTable.load, symmetric=True)
        self._pp_load_elements_output(element=_PpTable.ward, symmetric=True)
        self._pp_load_elements_output(element=_PpTable.motor, symmetric=True)
        self._pp_shunts_output()
        self._pp_trafos_output()
        self._pp_sgens_output()
        self._pp_trafos3w_output()
        self._pp_asym_gens_output()
        self._pp_asym_loads_output()
        # Switches derive results from branches pp_output_data and pgm_output_data of links. Hence, placed in the end.
        self._pp_switches_output()

    def _create_output_data_3ph(self):
        """
        Performs the conversion from power-grid-model to PandaPower by calling individual conversion functions.
        Furthermore, creates a global node lookup table, which stores nodes' voltage magnitude per unit and the voltage
        angle in degrees
        """
        # TODO create output_data_3ph for trafos3w, switches
        self._pp_buses_output_3ph()
        self._pp_lines_output_3ph()
        self._pp_ext_grids_output_3ph()
        self._pp_load_elements_output(element=_PpTable.load, symmetric=False)
        self._pp_load_elements_output(element=_PpTable.ward, symmetric=False)
        self._pp_load_elements_output(element=_PpTable.motor, symmetric=False)
        self._pp_shunts_output_3ph()
        self._pp_trafos_output_3ph()
        self._pp_sgens_output_3ph()
        self._pp_asym_gens_output_3ph()
        self._pp_asym_loads_output_3ph()

    def _create_pgm_input_nodes(self):
        """
        This function converts a Bus Dataframe of PandaPower to a power-grid-model Node input array.

        Returns:
            a power-grid-model structured array for the Node component
        """
        # TODO handle out-of-service buses, either here or in get_switch_states
        pp_busses = self.pp_input_data[_PpTable.bus]

        if pp_busses.empty:
            return

        if ComponentType.node in self.pgm_input_data:
            raise ValueError("Node component already exists in pgm_input_data")

        pgm_nodes = initialize_array(
            data_type=DatasetType.input, component_type=ComponentType.node, shape=len(pp_busses)
        )
        pgm_nodes[AttributeType.id] = self._generate_ids(_PpTable.bus, pp_busses.index)
        pgm_nodes[AttributeType.u_rated] = self._get_pp_attr(_PpTable.bus, _PpAttr.vn_kv, expected_type="f8") * 1e3

        self.pgm_input_data[ComponentType.node] = pgm_nodes

    def _create_pgm_input_lines(self):
        """
        This function converts a Line Dataframe of PandaPower to a power-grid-model Line input array.

        Returns:
            a power-grid-model structured array for the Line component
        """
        pp_lines = self.pp_input_data[_PpTable.line]

        if pp_lines.empty:
            return

        if ComponentType.line in self.pgm_input_data:
            raise ValueError("Line component already exists in pgm_input_data")

        switch_states = self.get_switch_states(_PpTable.line)
        in_service = self._get_pp_attr(_PpTable.line, _PpAttr.in_service, expected_type="bool", default=True)
        length_km = self._get_pp_attr(_PpTable.line, _PpAttr.length_km, expected_type="f8")
        parallel = self._get_pp_attr(_PpTable.line, _PpAttr.parallel, expected_type="u4", default=1)
        c_nf_per_km = self._get_pp_attr(_PpTable.line, _PpAttr.c_nf_per_km, expected_type="f8", default=0)
        c0_nf_per_km = self._get_pp_attr(_PpTable.line, _PpAttr.c0_nf_per_km, expected_type="f8", default=0)
        multiplier = length_km / parallel

        pgm_lines = initialize_array(
            data_type=DatasetType.input, component_type=ComponentType.line, shape=len(pp_lines)
        )
        pgm_lines[AttributeType.id] = self._generate_ids(_PpTable.line, pp_lines.index)
        pgm_lines[AttributeType.from_node] = self._get_pgm_ids(
            _PpTable.bus, self._get_pp_attr(_PpTable.line, _PpAttr.from_bus, expected_type="u4")
        )
        pgm_lines[AttributeType.from_status] = in_service & switch_states["from"]
        pgm_lines[AttributeType.to_node] = self._get_pgm_ids(
            _PpTable.bus, self._get_pp_attr(_PpTable.line, _PpAttr.to_bus, expected_type="u4")
        )
        pgm_lines[AttributeType.to_status] = in_service & switch_states["to"]
        pgm_lines[AttributeType.r1] = (
            self._get_pp_attr(_PpTable.line, _PpAttr.r_ohm_per_km, expected_type="f8") * multiplier
        )
        pgm_lines[AttributeType.x1] = (
            self._get_pp_attr(_PpTable.line, _PpAttr.x_ohm_per_km, expected_type="f8") * multiplier
        )
        pgm_lines[AttributeType.c1] = c_nf_per_km * length_km * parallel * 1e-9
        # The formula for tan1 = R_1 / Xc_1 = (g * 1e-6) / (2 * pi * f * c * 1e-9) = g / (2 * pi * f * c * 1e-3)
        pgm_lines[AttributeType.tan1] = np.divide(
            self._get_pp_attr(_PpTable.line, _PpAttr.g_us_per_km, expected_type="f8", default=0),
            c_nf_per_km * (2 * np.pi * self.system_frequency * 1e-3),
            where=np.logical_not(np.isclose(c_nf_per_km, 0.0)),
            out=None,
        )
        pgm_lines[AttributeType.i_n] = (
            (self._get_pp_attr(_PpTable.line, _PpAttr.max_i_ka, expected_type="f8", default=np.nan) * 1e3)
            * self._get_pp_attr(_PpTable.line, _PpAttr.df, expected_type="f8", default=1)
            * parallel
        )
        pgm_lines[AttributeType.r0] = (
            self._get_pp_attr(_PpTable.line, _PpAttr.r0_ohm_per_km, expected_type="f8", default=np.nan) * multiplier
        )
        pgm_lines[AttributeType.x0] = (
            self._get_pp_attr(_PpTable.line, _PpAttr.x0_ohm_per_km, expected_type="f8", default=np.nan) * multiplier
        )
        pgm_lines[AttributeType.c0] = c0_nf_per_km * length_km * parallel * 1e-9
        pgm_lines[AttributeType.tan0] = np.divide(
            self._get_pp_attr(_PpTable.line, _PpAttr.g0_us_per_km, expected_type="f8", default=0),
            c0_nf_per_km * (2 * np.pi * self.system_frequency * 1e-3),
            where=np.logical_not(np.isclose(c0_nf_per_km, 0.0)),
            out=None,
        )

        self.pgm_input_data[ComponentType.line] = pgm_lines

    def _create_pgm_input_sources(self):
        """
        This function converts External Grid Dataframe of PandaPower to a power-grid-model Source input array.

        Returns:
            a power-grid-model structured array for the Source component
        """
        pp_ext_grid = self.pp_input_data[_PpTable.ext_grid]

        if pp_ext_grid.empty:
            return

        if ComponentType.source in self.pgm_input_data:
            raise ValueError("Source component already exists in pgm_input_data")

        rx_max = self._get_pp_attr(_PpTable.ext_grid, _PpAttr.rx_max, expected_type="f8", default=np.nan)
        r0x0_max = self._get_pp_attr(_PpTable.ext_grid, _PpAttr.r0x0_max, expected_type="f8", default=np.nan)
        x0x_max = self._get_pp_attr(_PpTable.ext_grid, _PpAttr.x0x_max, expected_type="f8", default=np.nan)

        # Source Asym parameter check
        checks = {
            _PpAttr.r0x0_max: np.isnan(r0x0_max).all() or np.array_equal(rx_max, r0x0_max),
            _PpAttr.x0x_max: np.isnan(x0x_max).all() or all(x0x_max == 1),
        }
        if not all(checks.values()):
            failed_checks = ", ".join([key for key, value in checks.items() if not value])
            logger.warning("Zero sequence parameters given in external grid shall be ignored: %s", failed_checks)

        pgm_sources = initialize_array(
            data_type=DatasetType.input, component_type=ComponentType.source, shape=len(pp_ext_grid)
        )
        pgm_sources[AttributeType.id] = self._generate_ids(_PpTable.ext_grid, pp_ext_grid.index)
        pgm_sources[AttributeType.node] = self._get_pgm_ids(
            _PpTable.bus, self._get_pp_attr(_PpTable.ext_grid, _PpAttr.bus, expected_type="u4")
        )
        pgm_sources[AttributeType.status] = self._get_pp_attr(
            _PpTable.ext_grid, _PpAttr.in_service, expected_type="bool", default=True
        )
        pgm_sources[AttributeType.u_ref] = self._get_pp_attr(
            _PpTable.ext_grid, _PpAttr.vm_pu, expected_type="f8", default=1.0
        )
        pgm_sources[AttributeType.rx_ratio] = rx_max
        pgm_sources[AttributeType.u_ref_angle] = self._get_pp_attr(
            _PpTable.ext_grid, _PpAttr.va_degree, expected_type="f8", default=0.0
        ) * (np.pi / 180)
        pgm_sources[AttributeType.sk] = (
            self._get_pp_attr(_PpTable.ext_grid, _PpAttr.s_sc_max_mva, expected_type="f8", default=np.nan) * 1e6
        )

        self.pgm_input_data[ComponentType.source] = pgm_sources

    def _create_pgm_input_shunts(self):
        """
        This function converts a Shunt Dataframe of PandaPower to a power-grid-model Shunt input array.

        Returns:
            a power-grid-model structured array for the Shunt component
        """
        pp_shunts = self.pp_input_data[_PpTable.shunt]

        if pp_shunts.empty:
            return

        if ComponentType.shunt in self.pgm_input_data:
            raise ValueError("Shunt component already exists in pgm_input_data")

        vn_kv = self._get_pp_attr(_PpTable.shunt, _PpAttr.vn_kv, expected_type="f8")
        vn_kv_2 = vn_kv * vn_kv

        step = self._get_pp_attr(_PpTable.shunt, _PpAttr.step, expected_type="u4", default=1)
        g1_shunt = self._get_pp_attr(_PpTable.shunt, _PpAttr.p_mw, expected_type="f8") * step / vn_kv_2
        b1_shunt = -self._get_pp_attr(_PpTable.shunt, _PpAttr.q_mvar, expected_type="f8") * step / vn_kv_2

        pgm_shunts = initialize_array(
            data_type=DatasetType.input, component_type=ComponentType.shunt, shape=len(pp_shunts)
        )
        pgm_shunts[AttributeType.id] = self._generate_ids(_PpTable.shunt, pp_shunts.index)
        pgm_shunts[AttributeType.node] = self._get_pgm_ids(
            _PpTable.bus, self._get_pp_attr(_PpTable.shunt, _PpAttr.bus, expected_type="u4")
        )
        pgm_shunts[AttributeType.status] = self._get_pp_attr(
            _PpTable.shunt, _PpAttr.in_service, expected_type="bool", default=True
        )
        pgm_shunts[AttributeType.g1] = g1_shunt
        pgm_shunts[AttributeType.b1] = b1_shunt
        pgm_shunts[AttributeType.g0] = g1_shunt
        pgm_shunts[AttributeType.b0] = b1_shunt

        self.pgm_input_data[ComponentType.shunt] = pgm_shunts

    def _create_pgm_input_sym_gens(self):
        """
        This function converts a Static Generator Dataframe of PandaPower to a power-grid-model
        Symmetrical Generator input array.

        Returns:
            a power-grid-model structured array for the Symmetrical Generator component
        """
        pp_sgens = self.pp_input_data[_PpTable.sgen]

        if pp_sgens.empty:
            return

        if ComponentType.sym_gen in self.pgm_input_data:
            raise ValueError("Symmetric generator component already exists in pgm_input_data")

        scaling = self._get_pp_attr(_PpTable.sgen, _PpAttr.scaling, expected_type="f8", default=1.0)

        pgm_sym_gens = initialize_array(
            data_type=DatasetType.input, component_type=ComponentType.sym_gen, shape=len(pp_sgens)
        )
        pgm_sym_gens[AttributeType.id] = self._generate_ids(_PpTable.sgen, pp_sgens.index)
        pgm_sym_gens[AttributeType.node] = self._get_pgm_ids(
            _PpTable.bus, self._get_pp_attr(_PpTable.sgen, _PpAttr.bus, expected_type="i8")
        )
        pgm_sym_gens[AttributeType.status] = self._get_pp_attr(
            _PpTable.sgen, _PpAttr.in_service, expected_type="bool", default=True
        )
        pgm_sym_gens[AttributeType.p_specified] = self._get_pp_attr(_PpTable.sgen, _PpAttr.p_mw, expected_type="f8") * (
            1e6 * scaling
        )
        pgm_sym_gens[AttributeType.q_specified] = self._get_pp_attr(
            _PpTable.sgen, _PpAttr.q_mvar, expected_type="f8", default=0.0
        ) * (1e6 * scaling)
        pgm_sym_gens[AttributeType.type] = LoadGenType.const_power

        self.pgm_input_data[ComponentType.sym_gen] = pgm_sym_gens

    def _create_pgm_input_asym_gens(self):
        """
        This function converts an Asymmetric Static Generator Dataframe of PandaPower to a power-grid-model
        Asymmetrical Generator input array.

        Returns:
            a power-grid-model structured array for the Asymmetrical Generator component
        """
        pp_asym_gens = self.pp_input_data[_PpTable.asymmetric_sgen]

        if pp_asym_gens.empty:
            return

        if ComponentType.asym_gen in self.pgm_input_data:
            raise ValueError("Asymmetric generator component already exists in pgm_input_data")

        scaling = self._get_pp_attr(_PpTable.asymmetric_sgen, _PpAttr.scaling, expected_type="f8")
        multiplier = 1e6 * scaling

        pgm_asym_gens = initialize_array(
            data_type=DatasetType.input, component_type=ComponentType.asym_gen, shape=len(pp_asym_gens)
        )
        pgm_asym_gens[AttributeType.id] = self._generate_ids(_PpTable.asymmetric_sgen, pp_asym_gens.index)
        pgm_asym_gens[AttributeType.node] = self._get_pgm_ids(
            _PpTable.bus,
            self._get_pp_attr(_PpTable.asymmetric_sgen, _PpAttr.bus, expected_type="i8"),
        )
        pgm_asym_gens[AttributeType.status] = self._get_pp_attr(
            _PpTable.asymmetric_sgen, _PpAttr.in_service, expected_type="bool", default=True
        )
        pgm_asym_gens[AttributeType.p_specified] = np.transpose(
            np.array(
                (
                    self._get_pp_attr(_PpTable.asymmetric_sgen, _PpAttr.p_a_mw, expected_type="f8"),
                    self._get_pp_attr(_PpTable.asymmetric_sgen, _PpAttr.p_b_mw, expected_type="f8"),
                    self._get_pp_attr(_PpTable.asymmetric_sgen, _PpAttr.p_c_mw, expected_type="f8"),
                )
            )
            * multiplier
        )
        pgm_asym_gens[AttributeType.q_specified] = np.transpose(
            np.array(
                (
                    self._get_pp_attr(_PpTable.asymmetric_sgen, _PpAttr.q_a_mvar, expected_type="f8"),
                    self._get_pp_attr(_PpTable.asymmetric_sgen, _PpAttr.q_b_mvar, expected_type="f8"),
                    self._get_pp_attr(_PpTable.asymmetric_sgen, _PpAttr.q_c_mvar, expected_type="f8"),
                )
            )
            * multiplier
        )
        pgm_asym_gens[AttributeType.type] = LoadGenType.const_power

        self.pgm_input_data[ComponentType.asym_gen] = pgm_asym_gens

    def _create_pgm_input_sym_loads(self):
        """
        This function converts a Load Dataframe of PandaPower to a power-grid-model
        Symmetrical Load input array. For one load in PandaPower there are three loads in
        power-grid-model created.

        Returns:
            a power-grid-model structured array for the Symmetrical Load component
        """
        pp_loads = self.pp_input_data[_PpTable.load]

        if pp_loads.empty:
            return

        if ComponentType.sym_load in self.pgm_input_data:
            raise ValueError("Symmetrical Load component already exists in pgm_input_data")

        if np.any(self._get_pp_attr(_PpTable.load, _PpAttr.type, expected_type="O", default=None) == "delta"):
            raise NotImplementedError("Delta loads are not implemented, only wye loads are supported in PGM.")

        scaling = self._get_pp_attr(_PpTable.load, _PpAttr.scaling, expected_type="f8", default=1.0)
        in_service = self._get_pp_attr(_PpTable.load, _PpAttr.in_service, expected_type="bool", default=True)
        p_mw = self._get_pp_attr(_PpTable.load, _PpAttr.p_mw, expected_type="f8", default=0.0)
        q_mvar = self._get_pp_attr(_PpTable.load, _PpAttr.q_mvar, expected_type="f8", default=0.0)
        bus = self._get_pp_attr(_PpTable.load, _PpAttr.bus, expected_type="u4")

        n_loads = len(pp_loads)

        pgm_sym_loads = initialize_array(
            data_type=DatasetType.input, component_type=ComponentType.sym_load, shape=3 * n_loads
        )

        if PP_CONVERSION_VERSION < PP_COMPATIBILITY_VERSION_3_2_0:
            const_i_p_multiplier = (
                self._get_pp_attr(_PpTable.load, _PpAttr.const_i_percent, expected_type="f8", default=0)
                * scaling
                * (1e-2 * 1e6)
            )
            const_z_p_multiplier = (
                self._get_pp_attr(_PpTable.load, _PpAttr.const_z_percent, expected_type="f8", default=0)
                * scaling
                * (1e-2 * 1e6)
            )
            const_p_multiplier = (1e6 - const_i_p_multiplier - const_z_p_multiplier) * scaling
            const_q_multiplier = const_p_multiplier
            const_i_q_multiplier = const_i_p_multiplier
            const_z_q_multiplier = const_z_p_multiplier
        else:
            const_i_p_multiplier = (
                self._get_pp_attr(_PpTable.load, _PpAttr.const_i_p_percent, expected_type="f8", default=0)
                * scaling
                * (1e-2 * 1e6)
            )
            const_z_p_multiplier = (
                self._get_pp_attr(_PpTable.load, _PpAttr.const_z_p_percent, expected_type="f8", default=0)
                * scaling
                * (1e-2 * 1e6)
            )
            const_p_multiplier = (1e6 - const_i_p_multiplier - const_z_p_multiplier) * scaling
            const_i_q_multiplier = (
                self._get_pp_attr(_PpTable.load, _PpAttr.const_i_q_percent, expected_type="f8", default=0)
                * scaling
                * (1e-2 * 1e6)
            )
            const_z_q_multiplier = (
                self._get_pp_attr(_PpTable.load, _PpAttr.const_z_q_percent, expected_type="f8", default=0)
                * scaling
                * (1e-2 * 1e6)
            )
            const_q_multiplier = (1e6 - const_i_q_multiplier - const_z_q_multiplier) * scaling

        pgm_sym_loads[AttributeType.id][:n_loads] = self._generate_ids(
            _PpTable.load, pp_loads.index, name="const_power"
        )
        pgm_sym_loads[AttributeType.node][:n_loads] = self._get_pgm_ids(_PpTable.bus, bus)
        pgm_sym_loads[AttributeType.status][:n_loads] = in_service
        pgm_sym_loads[AttributeType.type][:n_loads] = LoadGenType.const_power
        pgm_sym_loads[AttributeType.p_specified][:n_loads] = const_p_multiplier * p_mw
        pgm_sym_loads[AttributeType.q_specified][:n_loads] = const_q_multiplier * q_mvar

        pgm_sym_loads[AttributeType.id][n_loads : 2 * n_loads] = self._generate_ids(
            _PpTable.load, pp_loads.index, name="const_impedance"
        )
        pgm_sym_loads[AttributeType.node][n_loads : 2 * n_loads] = self._get_pgm_ids(_PpTable.bus, bus)
        pgm_sym_loads[AttributeType.status][n_loads : 2 * n_loads] = in_service
        pgm_sym_loads[AttributeType.type][n_loads : 2 * n_loads] = LoadGenType.const_impedance
        pgm_sym_loads[AttributeType.p_specified][n_loads : 2 * n_loads] = const_z_p_multiplier * p_mw
        pgm_sym_loads[AttributeType.q_specified][n_loads : 2 * n_loads] = const_z_q_multiplier * q_mvar

        pgm_sym_loads[AttributeType.id][-n_loads:] = self._generate_ids(
            _PpTable.load, pp_loads.index, name="const_current"
        )
        pgm_sym_loads[AttributeType.node][-n_loads:] = self._get_pgm_ids(_PpTable.bus, bus)
        pgm_sym_loads[AttributeType.status][-n_loads:] = in_service
        pgm_sym_loads[AttributeType.type][-n_loads:] = LoadGenType.const_current
        pgm_sym_loads[AttributeType.p_specified][-n_loads:] = const_i_p_multiplier * p_mw
        pgm_sym_loads[AttributeType.q_specified][-n_loads:] = const_i_q_multiplier * q_mvar

        self.pgm_input_data[ComponentType.sym_load] = pgm_sym_loads

    def _create_pgm_input_asym_loads(self):
        """
        This function converts an asymmetric_load Dataframe of PandaPower to a power-grid-model asym_load input array.

        Returns:
            a power-grid-model structured array for the asym_load component
        """
        pp_asym_loads = self.pp_input_data[_PpTable.asymmetric_load]

        if pp_asym_loads.empty:
            return

        if ComponentType.asym_load in self.pgm_input_data:
            raise ValueError("Asymmetric Load component already exists in pgm_input_data")

        if np.any(
            self._get_pp_attr(_PpTable.asymmetric_load, _PpAttr.type, expected_type="O", default=None) == "delta"
        ):
            raise NotImplementedError("Delta loads are not implemented, only wye loads are supported in PGM.")

        scaling = self._get_pp_attr(_PpTable.asymmetric_load, _PpAttr.scaling, expected_type="f8")
        multiplier = 1e6 * scaling

        pgm_asym_loads = initialize_array(
            data_type=DatasetType.input, component_type=ComponentType.asym_load, shape=len(pp_asym_loads)
        )
        pgm_asym_loads[AttributeType.id] = self._generate_ids(_PpTable.asymmetric_load, pp_asym_loads.index)
        pgm_asym_loads[AttributeType.node] = self._get_pgm_ids(
            _PpTable.bus,
            self._get_pp_attr(_PpTable.asymmetric_load, _PpAttr.bus, expected_type="u4"),
        )
        pgm_asym_loads[AttributeType.status] = self._get_pp_attr(
            _PpTable.asymmetric_load, _PpAttr.in_service, expected_type="bool", default=True
        )
        pgm_asym_loads[AttributeType.p_specified] = np.transpose(
            np.array(
                [
                    self._get_pp_attr(_PpTable.asymmetric_load, _PpAttr.p_a_mw, expected_type="f8"),
                    self._get_pp_attr(_PpTable.asymmetric_load, _PpAttr.p_b_mw, expected_type="f8"),
                    self._get_pp_attr(_PpTable.asymmetric_load, _PpAttr.p_c_mw, expected_type="f8"),
                ]
            )
            * multiplier
        )
        pgm_asym_loads[AttributeType.q_specified] = np.transpose(
            np.array(
                [
                    self._get_pp_attr(_PpTable.asymmetric_load, _PpAttr.q_a_mvar, expected_type="f8"),
                    self._get_pp_attr(_PpTable.asymmetric_load, _PpAttr.q_b_mvar, expected_type="f8"),
                    self._get_pp_attr(_PpTable.asymmetric_load, _PpAttr.q_c_mvar, expected_type="f8"),
                ]
            )
            * multiplier
        )
        pgm_asym_loads[AttributeType.type] = LoadGenType.const_power

        self.pgm_input_data[ComponentType.asym_load] = pgm_asym_loads

    def _create_pgm_input_transformers(self):  # noqa: PLR0915  # pylint: disable=too-many-statements, disable-msg=too-many-locals
        """
        This function converts a Transformer Dataframe of PandaPower to a power-grid-model
        Transformer input array.

        Returns:
            a power-grid-model structured array for the Transformer component
        """
        pp_trafo = self.pp_input_data[_PpTable.trafo]

        if pp_trafo.empty:
            return

        if ComponentType.transformer in self.pgm_input_data:
            raise ValueError("Transformer component already exists in pgm_input_data")

        # Check for unsupported pandapower features
        if "tap_dependent_impedance" in pp_trafo.columns and any(pp_trafo["tap_dependent_impedance"]):
            raise RuntimeError("Tap dependent impedance is not supported in Power Grid Model")

        # Attribute retrieval
        i_no_load = self._get_pp_attr(_PpTable.trafo, _PpAttr.i0_percent, expected_type="f8")
        pfe = self._get_pp_attr(_PpTable.trafo, _PpAttr.pfe_kw, expected_type="f8")
        vk_percent = self._get_pp_attr(_PpTable.trafo, _PpAttr.vk_percent, expected_type="f8")
        vkr_percent = self._get_pp_attr(_PpTable.trafo, _PpAttr.vkr_percent, expected_type="f8")
        in_service = self._get_pp_attr(_PpTable.trafo, _PpAttr.in_service, expected_type="bool", default=True)
        parallel = self._get_pp_attr(_PpTable.trafo, _PpAttr.parallel, expected_type="u4", default=1)
        sn_mva = self._get_pp_attr(_PpTable.trafo, _PpAttr.sn_mva, expected_type="f8")
        switch_states = self.get_switch_states(_PpTable.trafo)

        tap_side = self._get_pp_attr(_PpTable.trafo, _PpAttr.tap_side, expected_type="O", default=None)
        tap_nom = self._get_pp_attr(_PpTable.trafo, _PpAttr.tap_neutral, expected_type="f8", default=np.nan)
        tap_pos = self._get_pp_attr(_PpTable.trafo, _PpAttr.tap_pos, expected_type="f8", default=np.nan)
        tap_size = self._get_tap_size(pp_trafo)
        winding_types = self.get_trafo_winding_types()
        clocks = (
            np.round(self._get_pp_attr(_PpTable.trafo, _PpAttr.shift_degree, expected_type="f8", default=0.0) / 30) % 12
        )

        # Asym parameters retrival and check. For PGM,
        # manual zero sequence vk0_percent and vkr0_percent params are not supported yet.
        vk0_percent = self._get_pp_attr(_PpTable.trafo, _PpAttr.vk0_percent, expected_type="f8", default=np.nan)
        vkr0_percent = self._get_pp_attr(_PpTable.trafo, _PpAttr.vkr0_percent, expected_type="f8", default=np.nan)
        # mag0_percent and mag0_rx will be fetched relative to vk_percent
        mag0_percent = self._get_pp_attr(_PpTable.trafo, _PpAttr.mag0_percent, expected_type="f8", default=np.nan)
        if PP_CONVERSION_VERSION < PP_COMPATIBILITY_VERSION_3_4_0:
            # before pandapower 3.4.0, the mag0_percent wasn't a percentage but a relative value between 0 and 1
            mag0_percent *= 100.0
        mag0_rx = self._get_pp_attr(_PpTable.trafo, _PpAttr.mag0_rx, expected_type="f8", default=np.nan)
        # Calculate rx ratio of magnetising branch
        valid = np.logical_and(np.not_equal(sn_mva, 0.0), np.isfinite(sn_mva))
        mag_g = np.divide(pfe, sn_mva * 1000, where=valid, out=None)
        mag_g[np.logical_not(valid)] = np.nan
        z_squared = i_no_load * i_no_load * 1e-4 - mag_g * mag_g
        valid = np.logical_and(np.greater(z_squared, 0), np.isfinite(z_squared))
        rx_mag = np.divide(mag_g, np.sqrt(z_squared, where=valid, out=None), where=valid, out=None)
        rx_mag[np.logical_not(valid)] = np.inf
        # positive and zero sequence magnetising impedance must be equal.
        # mag0_percent = z0mag / z0.
        checks = {
            "vk0_percent": np.logical_or(np.allclose(vk_percent, vk0_percent), np.isnan(vk0_percent).all()),
            "vkr0_percent": np.logical_or(np.allclose(vkr_percent, vkr0_percent), np.isnan(vkr0_percent).all()),
            "si0_hv_partial": np.isnan(
                self._get_pp_attr(_PpTable.trafo, _PpAttr.si0_hv_partial, expected_type="f8", default=np.nan)
            ).all(),
        }
        if not all(checks.values()):
            failed_checks = ", ".join([key for key, value in checks.items() if not value])
            logger.warning("Zero sequence parameters given in trafo shall be ignored: %s", failed_checks)
        valid = np.logical_and(
            np.not_equal(mag0_rx, 0.0),
            np.logical_and(
                np.allclose(vk_percent, vk0_percent),
                np.logical_and(
                    np.not_equal(vk_percent * mag0_percent * 1e-4, 0.0),
                    np.logical_and(np.logical_not(np.isnan(mag0_percent)), np.logical_not(np.isnan(mag0_rx))),
                ),
            ),
        )
        i0_zero_sequence = np.divide(
            np.ones_like(mag0_percent), (vk_percent * mag0_percent * 1e-4), out=None, where=valid
        )
        i0_zero_sequence[np.logical_not(valid)] = np.nan
        p0_zero_sequence = (
            np.divide(
                i0_zero_sequence,
                np.sqrt(1 + np.square(np.divide(np.ones_like(mag0_rx), mag0_rx, out=None, where=valid))),
                out=None,
                where=valid,
            )
            * sn_mva
            * 1e6
        )
        p0_zero_sequence[np.logical_not(valid)] = np.nan

        # Do not use taps when mandatory tap data is not available
        no_taps = np.equal(tap_side, None) | np.isnan(tap_pos) | np.isnan(tap_nom) | np.isnan(tap_size)
        tap_nom[no_taps] = 0
        tap_pos[no_taps] = 0
        tap_size[no_taps] = 0
        tap_side[no_taps] = "hv"

        # Default vector group for odd clocks = DYn and for even clocks = YNyn
        no_vector_groups = np.isnan(winding_types["winding_from"]) | np.isnan(winding_types["winding_to"])
        no_vector_groups_dyn = no_vector_groups & (clocks % 2)
        winding_types.loc[no_vector_groups] = WindingType.wye_n
        winding_types.loc[no_vector_groups_dyn, "winding_from"] = WindingType.delta

        # Create PGM array
        pgm_transformers = initialize_array(
            data_type=DatasetType.input, component_type=ComponentType.transformer, shape=len(pp_trafo)
        )
        pgm_transformers[AttributeType.id] = self._generate_ids(_PpTable.trafo, pp_trafo.index)
        pgm_transformers[AttributeType.from_node] = self._get_pgm_ids(
            _PpTable.bus, self._get_pp_attr(_PpTable.trafo, _PpAttr.hv_bus, expected_type="u4")
        )
        pgm_transformers[AttributeType.from_status] = in_service & switch_states["from"].values
        pgm_transformers[AttributeType.to_node] = self._get_pgm_ids(
            _PpTable.bus, self._get_pp_attr(_PpTable.trafo, _PpAttr.lv_bus, expected_type="u4")
        )
        pgm_transformers[AttributeType.to_status] = in_service & switch_states["to"].values
        pgm_transformers[AttributeType.u1] = (
            self._get_pp_attr(_PpTable.trafo, _PpAttr.vn_hv_kv, expected_type="f8") * 1e3
        )
        pgm_transformers[AttributeType.u2] = (
            self._get_pp_attr(_PpTable.trafo, _PpAttr.vn_lv_kv, expected_type="f8") * 1e3
        )
        pgm_transformers[AttributeType.sn] = sn_mva * parallel * 1e6
        pgm_transformers[AttributeType.uk] = vk_percent * 1e-2
        pgm_transformers[AttributeType.pk] = vkr_percent * sn_mva * parallel * (1e6 * 1e-2)
        pgm_transformers[AttributeType.p0] = pfe * parallel * 1e3
        pgm_transformers[AttributeType.i0] = i_no_load * 1e-2
        i0_min_threshold = pgm_transformers[AttributeType.p0] / pgm_transformers[AttributeType.sn]
        if any(np.less(pgm_transformers[AttributeType.i0], i0_min_threshold)):
            logger.warning("Minimum value of i0_percent is clipped to p0/sn")
            pgm_transformers[AttributeType.i0] = np.clip(
                pgm_transformers[AttributeType.i0], a_min=i0_min_threshold, a_max=None
            )
        pgm_transformers[AttributeType.i0_zero_sequence] = i0_zero_sequence
        pgm_transformers[AttributeType.p0_zero_sequence] = p0_zero_sequence * parallel
        pgm_transformers[AttributeType.clock] = clocks
        pgm_transformers[AttributeType.winding_from] = winding_types["winding_from"]
        pgm_transformers[AttributeType.winding_to] = winding_types["winding_to"]
        pgm_transformers[AttributeType.tap_nom] = tap_nom.astype("i4")
        pgm_transformers[AttributeType.tap_pos] = tap_pos.astype("i4")
        pgm_transformers[AttributeType.tap_side] = self._get_transformer_tap_side(tap_side)
        pgm_transformers[AttributeType.tap_min] = self._get_pp_attr(
            _PpTable.trafo, _PpAttr.tap_min, expected_type="i4", default=0
        )
        pgm_transformers[AttributeType.tap_max] = self._get_pp_attr(
            _PpTable.trafo, _PpAttr.tap_max, expected_type="i4", default=0
        )
        pgm_transformers[AttributeType.tap_size] = tap_size

        self.pgm_input_data[ComponentType.transformer] = pgm_transformers

    def _create_pgm_input_three_winding_transformers(self):  # noqa: PLR0915
        # pylint: disable=too-many-statements, disable-msg=too-many-locals
        """
        This function converts a Three Winding Transformer Dataframe of PandaPower to a power-grid-model
        Three Winding Transformer input array.

        Returns:
            a power-grid-model structured array for the Three Winding Transformer component
        """
        pp_trafo3w = self.pp_input_data[_PpTable.trafo3w]

        if pp_trafo3w.empty:
            return

        if ComponentType.three_winding_transformer in self.pgm_input_data:
            raise ValueError("Three-winding transformer component already exists in pgm_input_data")

        # Check for unsupported pandapower features
        if "tap_dependent_impedance" in pp_trafo3w.columns and any(pp_trafo3w["tap_dependent_impedance"]):
            raise RuntimeError("Tap dependent impedance is not supported in Power Grid Model")  # pragma: no cover
        if "tap_at_star_point" in pp_trafo3w.columns and any(pp_trafo3w["tap_at_star_point"]):
            raise RuntimeError("Tap at star point is not supported in Power Grid Model")

        # Attributes retrieval
        sn_hv_mva = self._get_pp_attr(_PpTable.trafo3w, _PpAttr.sn_hv_mva, expected_type="f8")
        sn_mv_mva = self._get_pp_attr(_PpTable.trafo3w, _PpAttr.sn_mv_mva, expected_type="f8")
        sn_lv_mva = self._get_pp_attr(_PpTable.trafo3w, _PpAttr.sn_lv_mva, expected_type="f8")
        in_service = self._get_pp_attr(_PpTable.trafo3w, _PpAttr.in_service, expected_type="bool", default=True)
        switch_states = self.get_trafo3w_switch_states(pp_trafo3w)
        tap_side = self._get_pp_attr(_PpTable.trafo3w, _PpAttr.tap_side, expected_type="O", default=None)
        tap_nom = self._get_pp_attr(_PpTable.trafo3w, _PpAttr.tap_neutral, expected_type="f8", default=np.nan)
        tap_pos = self._get_pp_attr(_PpTable.trafo3w, _PpAttr.tap_pos, expected_type="f8", default=np.nan)
        tap_size = self._get_3wtransformer_tap_size(pp_trafo3w)
        vk_hv_percent = self._get_pp_attr(_PpTable.trafo3w, _PpAttr.vk_hv_percent, expected_type="f8")
        vkr_hv_percent = self._get_pp_attr(_PpTable.trafo3w, _PpAttr.vkr_hv_percent, expected_type="f8")
        vk_mv_percent = self._get_pp_attr(_PpTable.trafo3w, _PpAttr.vk_mv_percent, expected_type="f8")
        vkr_mv_percent = self._get_pp_attr(_PpTable.trafo3w, _PpAttr.vkr_mv_percent, expected_type="f8")
        vk_lv_percent = self._get_pp_attr(_PpTable.trafo3w, _PpAttr.vk_lv_percent, expected_type="f8")
        vkr_lv_percent = self._get_pp_attr(_PpTable.trafo3w, _PpAttr.vkr_lv_percent, expected_type="f8")
        winding_types = self.get_trafo3w_winding_types()
        clocks_12 = (
            np.round(
                self._get_pp_attr(_PpTable.trafo3w, _PpAttr.shift_mv_degree, expected_type="f8", default=0.0) / 30.0
            )
            % 12
        )
        clocks_13 = (
            np.round(
                self._get_pp_attr(_PpTable.trafo3w, _PpAttr.shift_lv_degree, expected_type="f8", default=0.0) / 30.0
            )
            % 12
        )
        vk0_hv_percent = self._get_pp_attr(_PpTable.trafo3w, _PpAttr.vk0_hv_percent, expected_type="f8", default=np.nan)
        vkr0_hv_percent = self._get_pp_attr(
            _PpTable.trafo3w, _PpAttr.vkr0_hv_percent, expected_type="f8", default=np.nan
        )
        vk0_mv_percent = self._get_pp_attr(_PpTable.trafo3w, _PpAttr.vk0_mv_percent, expected_type="f8", default=np.nan)
        vkr0_mv_percent = self._get_pp_attr(
            _PpTable.trafo3w, _PpAttr.vkr0_mv_percent, expected_type="f8", default=np.nan
        )
        vk0_lv_percent = self._get_pp_attr(_PpTable.trafo3w, _PpAttr.vk0_lv_percent, expected_type="f8", default=np.nan)
        vkr0_lv_percent = self._get_pp_attr(
            _PpTable.trafo3w, _PpAttr.vkr0_lv_percent, expected_type="f8", default=np.nan
        )

        # Asym parameters. For PGM, manual zero sequence params are not supported yet.
        checks = {
            _PpAttr.vk0_hv_percent: np.array_equal(vk_hv_percent, vk0_hv_percent) or np.isnan(vk0_hv_percent).all(),
            _PpAttr.vkr0_hv_percent: np.array_equal(vkr_hv_percent, vkr0_hv_percent) or np.isnan(vkr0_hv_percent).all(),
            _PpAttr.vk0_mv_percent: np.array_equal(vk_mv_percent, vk0_mv_percent) or np.isnan(vk0_mv_percent).all(),
            _PpAttr.vkr0_mv_percent: np.array_equal(vkr_mv_percent, vkr0_mv_percent) or np.isnan(vkr0_mv_percent).all(),
            _PpAttr.vk0_lv_percent: np.array_equal(vk_lv_percent, vk0_lv_percent) or np.isnan(vk0_lv_percent).all(),
            _PpAttr.vkr0_lv_percent: np.array_equal(vkr_lv_percent, vkr0_lv_percent) or np.isnan(vkr0_lv_percent).all(),
        }
        if not all(checks.values()):
            failed_checks = ", ".join([key for key, value in checks.items() if not value])
            logger.warning("Zero sequence parameters given in trafo3w are ignored: %s", failed_checks)

        # Do not use taps when mandatory tap data is not available
        no_taps = np.equal(tap_side, None) | np.isnan(tap_pos) | np.isnan(tap_nom) | np.isnan(tap_size)
        tap_nom[no_taps] = 0
        tap_pos[no_taps] = 0
        tap_size[no_taps] = 0
        tap_side[no_taps] = "hv"

        # Default vector group for odd clocks_12 = Yndx, for odd clocks_13 = Ynxd and for even clocks = YNxyn or YNynx
        no_vector_groups = (
            np.isnan(winding_types["winding_1"])
            & np.isnan(winding_types["winding_2"])
            & np.isnan(winding_types["winding_3"])
        )
        no_vector_groups_ynd2 = no_vector_groups & (clocks_12 % 2)
        no_vector_groups_ynd3 = no_vector_groups & (clocks_13 % 2)
        winding_types[no_vector_groups] = WindingType.wye_n
        winding_types.loc[no_vector_groups_ynd2, "winding_2"] = WindingType.delta
        winding_types.loc[no_vector_groups_ynd3, "winding_3"] = WindingType.delta

        pgm_3wtransformers = initialize_array(
            data_type=DatasetType.input,
            component_type=ComponentType.three_winding_transformer,
            shape=len(pp_trafo3w),
        )
        pgm_3wtransformers[AttributeType.id] = self._generate_ids(_PpTable.trafo3w, pp_trafo3w.index)

        pgm_3wtransformers[AttributeType.node_1] = self._get_pgm_ids(
            _PpTable.bus, self._get_pp_attr(_PpTable.trafo3w, _PpAttr.hv_bus, expected_type="u4")
        )
        pgm_3wtransformers[AttributeType.node_2] = self._get_pgm_ids(
            _PpTable.bus, self._get_pp_attr(_PpTable.trafo3w, _PpAttr.mv_bus, expected_type="u4")
        )
        pgm_3wtransformers[AttributeType.node_3] = self._get_pgm_ids(
            _PpTable.bus, self._get_pp_attr(_PpTable.trafo3w, _PpAttr.lv_bus, expected_type="u4")
        )
        pgm_3wtransformers[AttributeType.status_1] = in_service & switch_states["side_1"].values
        pgm_3wtransformers[AttributeType.status_2] = in_service & switch_states["side_2"].values
        pgm_3wtransformers[AttributeType.status_3] = in_service & switch_states["side_3"].values
        pgm_3wtransformers[AttributeType.u1] = (
            self._get_pp_attr(_PpTable.trafo3w, _PpAttr.vn_hv_kv, expected_type="f8") * 1e3
        )
        pgm_3wtransformers[AttributeType.u2] = (
            self._get_pp_attr(_PpTable.trafo3w, _PpAttr.vn_mv_kv, expected_type="f8") * 1e3
        )
        pgm_3wtransformers[AttributeType.u3] = (
            self._get_pp_attr(_PpTable.trafo3w, _PpAttr.vn_lv_kv, expected_type="f8") * 1e3
        )
        pgm_3wtransformers[AttributeType.sn_1] = sn_hv_mva * 1e6
        pgm_3wtransformers[AttributeType.sn_2] = sn_mv_mva * 1e6
        pgm_3wtransformers[AttributeType.sn_3] = sn_lv_mva * 1e6
        pgm_3wtransformers[AttributeType.uk_12] = vk_hv_percent * 1e-2
        pgm_3wtransformers[AttributeType.uk_13] = vk_lv_percent * 1e-2
        pgm_3wtransformers[AttributeType.uk_23] = vk_mv_percent * 1e-2

        pgm_3wtransformers[AttributeType.pk_12] = vkr_hv_percent * np.minimum(sn_hv_mva, sn_mv_mva) * (1e-2 * 1e6)
        pgm_3wtransformers[AttributeType.pk_13] = vkr_lv_percent * np.minimum(sn_hv_mva, sn_lv_mva) * (1e-2 * 1e6)
        pgm_3wtransformers[AttributeType.pk_23] = vkr_mv_percent * np.minimum(sn_mv_mva, sn_lv_mva) * (1e-2 * 1e6)

        pgm_3wtransformers[AttributeType.p0] = (
            self._get_pp_attr(_PpTable.trafo3w, _PpAttr.pfe_kw, expected_type="f8") * 1e3
        )
        pgm_3wtransformers[AttributeType.i0] = (
            self._get_pp_attr(_PpTable.trafo3w, _PpAttr.i0_percent, expected_type="f8") * 1e-2
        )
        i0_min_threshold = pgm_3wtransformers[AttributeType.p0] / pgm_3wtransformers[AttributeType.sn_1]
        if any(np.less(pgm_3wtransformers[AttributeType.i0], i0_min_threshold)):
            logger.warning("Minimum value of i0_percent is clipped to p0/sn_1")
            pgm_3wtransformers[AttributeType.i0] = np.clip(
                pgm_3wtransformers[AttributeType.i0], a_min=i0_min_threshold, a_max=None
            )
        pgm_3wtransformers[AttributeType.clock_12] = clocks_12
        pgm_3wtransformers[AttributeType.clock_13] = clocks_13
        pgm_3wtransformers[AttributeType.winding_1] = winding_types["winding_1"]
        pgm_3wtransformers[AttributeType.winding_2] = winding_types["winding_2"]
        pgm_3wtransformers[AttributeType.winding_3] = winding_types["winding_3"]
        pgm_3wtransformers[AttributeType.tap_nom] = tap_nom.astype("i4")  # TODO(mgovers) shouldn't this be rounded?
        pgm_3wtransformers[AttributeType.tap_pos] = tap_pos.astype("i4")  # TODO(mgovers) shouldn't this be rounded?
        pgm_3wtransformers[AttributeType.tap_side] = self._get_3wtransformer_tap_side(tap_side)
        pgm_3wtransformers[AttributeType.tap_min] = self._get_pp_attr(
            _PpTable.trafo3w, _PpAttr.tap_min, expected_type="i4", default=0
        )
        pgm_3wtransformers[AttributeType.tap_max] = self._get_pp_attr(
            _PpTable.trafo3w, _PpAttr.tap_max, expected_type="i4", default=0
        )
        pgm_3wtransformers[AttributeType.tap_size] = tap_size

        self.pgm_input_data[ComponentType.three_winding_transformer] = pgm_3wtransformers

    def _create_pgm_input_links(self):
        """
        This function takes a Switch Dataframe of PandaPower, extracts the Switches which have Bus to Bus
        connection and converts them to a power-grid-model Link input array.

        Returns:
            a power-grid-model structured array for the Link component
        """
        pp_switches = self.pp_input_data[_PpTable.switch]

        if pp_switches.empty:
            return

        if ComponentType.link in self.pgm_input_data:
            raise ValueError("Link component already exists in pgm_input_data")

        # This should take all the switches which are b2b
        pp_switches = pp_switches[pp_switches[_PpAttr.et] == "b"]

        pgm_links = initialize_array(
            data_type=DatasetType.input, component_type=ComponentType.link, shape=len(pp_switches)
        )
        pgm_links[AttributeType.id] = self._generate_ids(_PpTable.switch, pp_switches.index, name="b2b_switches")
        pgm_links[AttributeType.from_node] = self._get_pgm_ids(_PpTable.bus, pp_switches[_PpTable.bus])
        pgm_links[AttributeType.to_node] = self._get_pgm_ids(_PpTable.bus, pp_switches[_PpAttr.element])
        pgm_links[AttributeType.from_status] = pp_switches["closed"]
        pgm_links[AttributeType.to_status] = pp_switches["closed"]

        self.pgm_input_data[ComponentType.link] = pgm_links

    def _create_pgm_input_storages(self):
        # 3ph output to be made available too
        pp_storage = self.pp_input_data[_PpTable.storage]

        if pp_storage.empty:
            return

        raise NotImplementedError("Storage is not implemented yet!")

    def _create_pgm_input_impedances(self):
        pp_impedance = self.pp_input_data[_PpTable.impedance]

        if pp_impedance.empty:
            return

        raise NotImplementedError("Impedance is not implemented yet!")

    def _create_pgm_input_wards(self):
        pp_wards = self.pp_input_data[_PpTable.ward]

        if pp_wards.empty:
            return

        n_wards = len(pp_wards)
        in_service = self._get_pp_attr(_PpTable.ward, _PpAttr.in_service, expected_type="bool", default=True)
        bus = self._get_pp_attr(_PpTable.ward, _PpAttr.bus, expected_type="u4")

        pgm_sym_loads_from_ward = initialize_array(
            data_type=DatasetType.input, component_type=ComponentType.sym_load, shape=n_wards * 2
        )
        pgm_sym_loads_from_ward[AttributeType.id][:n_wards] = self._generate_ids(
            _PpTable.ward, pp_wards.index, name="ward_const_power_load"
        )
        pgm_sym_loads_from_ward[AttributeType.node][:n_wards] = self._get_pgm_ids(_PpTable.bus, bus)
        pgm_sym_loads_from_ward[AttributeType.status][:n_wards] = in_service
        pgm_sym_loads_from_ward[AttributeType.type][:n_wards] = LoadGenType.const_power
        pgm_sym_loads_from_ward[AttributeType.p_specified][:n_wards] = (
            self._get_pp_attr(_PpTable.ward, _PpAttr.ps_mw, expected_type="f8") * 1e6
        )
        pgm_sym_loads_from_ward[AttributeType.q_specified][:n_wards] = (
            self._get_pp_attr(_PpTable.ward, _PpAttr.qs_mvar, expected_type="f8") * 1e6
        )

        pgm_sym_loads_from_ward[AttributeType.id][-n_wards:] = self._generate_ids(
            _PpTable.ward, pp_wards.index, name="ward_const_impedance_load"
        )
        pgm_sym_loads_from_ward[AttributeType.node][-n_wards:] = self._get_pgm_ids(_PpTable.bus, bus)
        pgm_sym_loads_from_ward[AttributeType.status][-n_wards:] = in_service
        pgm_sym_loads_from_ward[AttributeType.type][-n_wards:] = LoadGenType.const_impedance
        pgm_sym_loads_from_ward[AttributeType.p_specified][-n_wards:] = (
            self._get_pp_attr(_PpTable.ward, _PpAttr.pz_mw, expected_type="f8") * 1e6
        )
        pgm_sym_loads_from_ward[AttributeType.q_specified][-n_wards:] = (
            self._get_pp_attr(_PpTable.ward, _PpAttr.qz_mvar, expected_type="f8") * 1e6
        )

        #  If input data of loads has already been filled then extend it with data of wards. If it is empty and there
        #  is no data about loads,then assign ward data to it
        if ComponentType.sym_load in self.pgm_input_data:
            symload_dtype = self.pgm_input_data[ComponentType.sym_load].dtype
            self.pgm_input_data[ComponentType.sym_load] = np.concatenate(  # pylint: disable=unexpected-keyword-arg
                [self.pgm_input_data[ComponentType.sym_load], pgm_sym_loads_from_ward],
                dtype=symload_dtype,
            )
        else:
            self.pgm_input_data[ComponentType.sym_load] = pgm_sym_loads_from_ward

    def _create_pgm_input_xwards(self):
        pp_xwards = self.pp_input_data[_PpTable.xward]

        if pp_xwards.empty:
            return

        raise NotImplementedError("Extended Ward is not implemented yet!")

    def _create_pgm_input_motors(self):
        pp_motors = self.pp_input_data[_PpTable.motor]

        if pp_motors.empty:
            return

        pgm_sym_loads_from_motor = initialize_array(
            data_type=DatasetType.input, component_type=ComponentType.sym_load, shape=len(pp_motors)
        )
        pgm_sym_loads_from_motor[AttributeType.id] = self._generate_ids(
            _PpTable.motor, pp_motors.index, name="motor_load"
        )
        pgm_sym_loads_from_motor[AttributeType.node] = self._get_pgm_ids(
            _PpTable.bus, self._get_pp_attr(_PpTable.motor, _PpAttr.bus, expected_type="i8")
        )
        pgm_sym_loads_from_motor[AttributeType.status] = self._get_pp_attr(
            _PpTable.motor, _PpAttr.in_service, expected_type="bool", default=True
        )
        pgm_sym_loads_from_motor[AttributeType.type] = LoadGenType.const_power
        #  The formula for p_specified is pn_mech_mw /(efficiency_percent/100) * (loading_percent/100) * scaling * 1e6
        pgm_sym_loads_from_motor[AttributeType.p_specified] = (
            self._get_pp_attr(_PpTable.motor, _PpAttr.pn_mech_mw, expected_type="f8")
            / self._get_pp_attr(_PpTable.motor, _PpAttr.efficiency_percent, expected_type="f8")
            * self._get_pp_attr(_PpTable.motor, _PpAttr.loading_percent, expected_type="f8")
            * self._get_pp_attr(_PpTable.motor, _PpAttr.scaling, expected_type="f8")
            * 1e6
        )
        p_spec = pgm_sym_loads_from_motor[AttributeType.p_specified]
        cos_phi = self._get_pp_attr(_PpTable.motor, _PpAttr.cos_phi, expected_type="f8")
        valid = np.logical_and(np.not_equal(cos_phi, 0.0), np.isfinite(cos_phi))
        q_spec = np.sqrt(
            np.power(np.divide(p_spec, cos_phi, where=valid, out=None), 2, where=valid, out=None) - p_spec**2,
            where=valid,
            out=None,
        )
        q_spec[np.logical_not(valid)] = np.nan
        pgm_sym_loads_from_motor[AttributeType.q_specified] = q_spec

        #  If input data of loads has already been filled then extend it with data of motors. If it is empty and there
        #  is no data about loads,then assign motor data to it
        if ComponentType.sym_load in self.pgm_input_data:
            symload_dtype = self.pgm_input_data[ComponentType.sym_load].dtype
            self.pgm_input_data[ComponentType.sym_load] = np.concatenate(  # pylint: disable=unexpected-keyword-arg
                [self.pgm_input_data[ComponentType.sym_load], pgm_sym_loads_from_motor],
                dtype=symload_dtype,
            )
        else:
            self.pgm_input_data[ComponentType.sym_load] = pgm_sym_loads_from_motor

    def _create_pgm_input_dclines(self):
        pp_dcline = self.pp_input_data[_PpTable.dcline]

        if pp_dcline.empty:
            return

        raise NotImplementedError("DC line is not implemented yet. power-grid-model does not support PV buses yet")

    def _create_pgm_input_generators(self):
        pp_gen = self.pp_input_data[_PpTable.gen]

        if pp_gen.empty:
            return

        raise NotImplementedError("Generators is not implemented yet. power-grid-model does not support PV buses yet")

    def _pp_buses_output(self):
        """
        This function converts a power-grid-model Node output array to a Bus Dataframe of PandaPower.

        Returns:
            a PandaPower Dataframe for the Bus component
        """
        if _PpTable.res_bus in self.pp_output_data:
            raise ValueError("res_bus already exists in pp_output_data.")

        if ComponentType.node not in self.pgm_output_data or self.pgm_output_data[ComponentType.node].size == 0:
            return

        pgm_nodes = self.pgm_output_data[ComponentType.node]

        pp_output_buses = pd.DataFrame(
            columns=[_PpAttr.vm_pu, _PpAttr.va_degree, _PpAttr.p_mw, _PpAttr.q_mvar],
            index=self._get_pp_ids(_PpTable.bus, pgm_nodes[AttributeType.id]),
        )

        pp_output_buses[_PpAttr.vm_pu] = pgm_nodes[AttributeType.u_pu]
        pp_output_buses[_PpAttr.va_degree] = pgm_nodes[AttributeType.u_angle] * (180.0 / np.pi)

        # p_to, p_from, q_to and q_from connected to the bus have to be summed up
        self._pp_buses_output__accumulate_power(pp_output_buses)

        self.pp_output_data[_PpTable.res_bus] = pp_output_buses

    def _pp_buses_output__accumulate_power(self, pp_output_buses: pd.DataFrame):
        """
        For each node, we accumulate the power for all connected branches and branch3s

        Args:
            pp_output_buses: a Pandapower output dataframe of Bus component

        Returns:
            accumulated power for each bus
        """

        # Let's define all the components and sides where nodes can be connected
        component_sides = {
            ComponentType.line: [
                (AttributeType.from_node, AttributeType.p_from, AttributeType.q_from),
                (AttributeType.to_node, AttributeType.p_to, AttributeType.q_to),
            ],
            ComponentType.link: [
                (AttributeType.from_node, AttributeType.p_from, AttributeType.q_from),
                (AttributeType.to_node, AttributeType.p_to, AttributeType.q_to),
            ],
            ComponentType.transformer: [
                (AttributeType.from_node, AttributeType.p_from, AttributeType.q_from),
                (AttributeType.to_node, AttributeType.p_to, AttributeType.q_to),
            ],
            ComponentType.three_winding_transformer: [
                (AttributeType.node_1, AttributeType.p_1, AttributeType.q_1),
                (AttributeType.node_2, AttributeType.p_2, AttributeType.q_2),
                (AttributeType.node_3, AttributeType.p_3, AttributeType.q_3),
            ],
        }

        # Set the initial powers to zero
        pp_output_buses[_PpAttr.p_mw] = 0.0
        pp_output_buses[_PpAttr.q_mvar] = 0.0

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
                accumulated_data.index = self._get_pp_ids(_PpTable.bus, pd.Series(accumulated_data.index))

                # We might not have power data for each pp bus, so select only the indexes for which data is available
                idx = pp_output_buses.index.intersection(accumulated_data.index)

                # Now add the active and reactive powers to the pp busses
                # Note that the units are incorrect; for efficiency, unit conversions will be applied at the end.
                pp_output_buses.loc[idx, _PpAttr.p_mw] -= accumulated_data[p_col]
                pp_output_buses.loc[idx, _PpAttr.q_mvar] -= accumulated_data[q_col]

        # Finally apply the unit conversion (W -> MW and VAR -> MVAR)
        pp_output_buses[_PpAttr.p_mw] /= 1e6
        pp_output_buses[_PpAttr.q_mvar] /= 1e6

    def _pp_lines_output(self):
        """
        This function converts a power-grid-model Line output array to a Line Dataframe of PandaPower.

        Returns:
            a PandaPower Dataframe for the Line component
        """
        if _PpTable.res_line in self.pp_output_data:
            raise ValueError("res_line already exists in pp_output_data.")

        if ComponentType.line not in self.pgm_output_data or self.pgm_output_data[ComponentType.line].size == 0:
            return

        pgm_input_lines = self.pgm_input_data[ComponentType.line]
        pgm_output_lines = self.pgm_output_data[ComponentType.line]

        if not np.array_equal(pgm_input_lines[AttributeType.id], pgm_output_lines[AttributeType.id]):
            raise ValueError("The output line ids should correspond to the input line ids")

        pp_output_lines = pd.DataFrame(
            columns=[
                _PpAttr.p_from_mw,
                _PpAttr.q_from_mvar,
                _PpAttr.p_to_mw,
                _PpAttr.q_to_mvar,
                _PpAttr.pl_mw,
                _PpAttr.ql_mvar,
                _PpAttr.i_from_ka,
                _PpAttr.i_to_ka,
                _PpAttr.i_ka,
                _PpAttr.vm_from_pu,
                _PpAttr.vm_to_pu,
                _PpAttr.va_from_degree,
                _PpAttr.va_to_degree,
                _PpAttr.loading_percent,
            ],
            index=self._get_pp_ids(ComponentType.line, pgm_output_lines[AttributeType.id]),
        )

        from_nodes = self.pgm_nodes_lookup.loc[pgm_input_lines[AttributeType.from_node]]
        to_nodes = self.pgm_nodes_lookup.loc[pgm_input_lines[AttributeType.to_node]]

        pp_output_lines[_PpAttr.p_from_mw] = pgm_output_lines[AttributeType.p_from] * 1e-6
        pp_output_lines[_PpAttr.q_from_mvar] = pgm_output_lines[AttributeType.q_from] * 1e-6
        pp_output_lines[_PpAttr.p_to_mw] = pgm_output_lines[AttributeType.p_to] * 1e-6
        pp_output_lines[_PpAttr.q_to_mvar] = pgm_output_lines[AttributeType.q_to] * 1e-6
        pp_output_lines[_PpAttr.pl_mw] = (
            pgm_output_lines[AttributeType.p_from] + pgm_output_lines[AttributeType.p_to]
        ) * 1e-6
        pp_output_lines[_PpAttr.ql_mvar] = (
            pgm_output_lines[AttributeType.q_from] + pgm_output_lines[AttributeType.q_to]
        ) * 1e-6
        pp_output_lines[_PpAttr.i_from_ka] = pgm_output_lines[AttributeType.i_from] * 1e-3
        pp_output_lines[_PpAttr.i_to_ka] = pgm_output_lines[AttributeType.i_to] * 1e-3
        pp_output_lines[_PpAttr.i_ka] = (
            np.maximum(pgm_output_lines[AttributeType.i_from], pgm_output_lines[AttributeType.i_to]) * 1e-3
        )
        pp_output_lines[_PpAttr.vm_from_pu] = from_nodes["u_pu"].values
        pp_output_lines[_PpAttr.vm_to_pu] = to_nodes["u_pu"].values
        pp_output_lines[_PpAttr.va_from_degree] = from_nodes["u_degree"].values
        pp_output_lines[_PpAttr.va_to_degree] = to_nodes["u_degree"].values
        pp_output_lines[_PpAttr.loading_percent] = pgm_output_lines[AttributeType.loading] * 1e2

        self.pp_output_data[_PpTable.res_line] = pp_output_lines

    def _pp_ext_grids_output(self):
        """
        This function converts a power-grid-model Source output array to an External Grid Dataframe of PandaPower.

        Returns:
            a PandaPower Dataframe for the External Grid component
        """
        if _PpTable.res_ext_grid in self.pp_output_data:
            raise ValueError("res_ext_grid already exists in pp_output_data.")

        if ComponentType.source not in self.pgm_output_data or self.pgm_output_data[ComponentType.source].size == 0:
            return

        pgm_output_sources = self.pgm_output_data[ComponentType.source]

        pp_output_ext_grids = pd.DataFrame(
            columns=[_PpAttr.p_mw, _PpAttr.q_mvar],
            index=self._get_pp_ids(_PpTable.ext_grid, pgm_output_sources[AttributeType.id]),
        )
        pp_output_ext_grids[_PpAttr.p_mw] = pgm_output_sources[AttributeType.p] * 1e-6
        pp_output_ext_grids[_PpAttr.q_mvar] = pgm_output_sources[AttributeType.q] * 1e-6

        self.pp_output_data[_PpTable.res_ext_grid] = pp_output_ext_grids

    def _pp_shunts_output(self):
        """
        This function converts a power-grid-model Shunt output array to a Shunt Dataframe of PandaPower.

        Returns:
            a PandaPower Dataframe for the Shunt component
        """
        if _PpTable.res_shunt in self.pp_output_data:
            raise ValueError("res_shunt already exists in pp_output_data.")

        if ComponentType.shunt not in self.pgm_output_data or self.pgm_output_data[ComponentType.shunt].size == 0:
            return

        pgm_input_shunts = self.pgm_input_data[ComponentType.shunt]

        pgm_output_shunts = self.pgm_output_data[ComponentType.shunt]

        at_nodes = self.pgm_nodes_lookup.loc[pgm_input_shunts[AttributeType.node]]

        pp_output_shunts = pd.DataFrame(
            columns=[_PpAttr.p_mw, _PpAttr.q_mvar, _PpAttr.vm_pu],
            index=self._get_pp_ids(_PpTable.shunt, pgm_output_shunts[AttributeType.id]),
        )
        pp_output_shunts[_PpAttr.p_mw] = pgm_output_shunts[AttributeType.p] * 1e-6
        pp_output_shunts[_PpAttr.q_mvar] = pgm_output_shunts[AttributeType.q] * 1e-6
        pp_output_shunts[_PpAttr.vm_pu] = at_nodes[AttributeType.u_pu].values

        self.pp_output_data[_PpTable.res_shunt] = pp_output_shunts

    def _pp_sgens_output(self):
        """
        This function converts a power-grid-model Symmetrical Generator output array to a Static Generator Dataframe of
        PandaPower.

        Returns:
            a PandaPower Dataframe for the Static Generator component
        """
        if _PpTable.res_sgen in self.pp_output_data:
            raise ValueError("res_sgen already exists in pp_output_data.")

        if ComponentType.sym_gen not in self.pgm_output_data or self.pgm_output_data[ComponentType.sym_gen].size == 0:
            return

        pgm_output_sym_gens = self.pgm_output_data[ComponentType.sym_gen]

        pp_output_sgens = pd.DataFrame(
            columns=[_PpAttr.p_mw, _PpAttr.q_mvar],
            index=self._get_pp_ids(_PpTable.sgen, pgm_output_sym_gens[AttributeType.id]),
        )
        pp_output_sgens[_PpAttr.p_mw] = pgm_output_sym_gens[AttributeType.p] * 1e-6
        pp_output_sgens[_PpAttr.q_mvar] = pgm_output_sym_gens[AttributeType.q] * 1e-6

        self.pp_output_data[_PpTable.res_sgen] = pp_output_sgens

    def _pp_trafos_output(self):
        """
        This function converts a power-grid-model Transformer output array to a Transformer Dataframe of
        PandaPower.

        Returns:
            a PandaPower Dataframe for the Transformer component
        """
        if _PpTable.res_trafo in self.pp_output_data:
            raise ValueError("res_trafo already exists in pp_output_data.")

        if (
            ComponentType.transformer not in self.pgm_output_data
            or self.pgm_output_data[ComponentType.transformer].size == 0
        ) or (_PpTable.trafo not in self.pp_input_data or len(self.pp_input_data[_PpTable.trafo]) == 0):
            return

        pgm_input_transformers = self.pgm_input_data[ComponentType.transformer]
        pp_input_transformers = self.pp_input_data[_PpTable.trafo]
        pgm_output_transformers = self.pgm_output_data[ComponentType.transformer]

        from_nodes = self.pgm_nodes_lookup.loc[pgm_input_transformers[AttributeType.from_node]]
        to_nodes = self.pgm_nodes_lookup.loc[pgm_input_transformers[AttributeType.to_node]]

        # Only derating factor used here. Sn is already being multiplied by parallel
        loading_multiplier = pp_input_transformers["df"]
        if self.trafo_loading == "current":
            ui_from = pgm_output_transformers[AttributeType.i_from] * pgm_input_transformers[AttributeType.u1]
            ui_to = pgm_output_transformers[AttributeType.i_to] * pgm_input_transformers[AttributeType.u2]
            loading = (
                (np.sqrt(3) * np.maximum(ui_from, ui_to) / pgm_input_transformers[AttributeType.sn])
                * loading_multiplier
                * 1e2
            )
        elif self.trafo_loading == "power":
            loading = pgm_output_transformers[AttributeType.loading] * loading_multiplier * 1e2
        else:
            raise ValueError(f"Invalid transformer loading type: {self.trafo_loading!s}")

        pp_output_trafos = pd.DataFrame(
            columns=[
                _PpAttr.p_hv_mw,
                _PpAttr.q_hv_mvar,
                _PpAttr.p_lv_mw,
                _PpAttr.q_lv_mvar,
                _PpAttr.pl_mw,
                _PpAttr.ql_mvar,
                _PpAttr.i_hv_ka,
                _PpAttr.i_lv_ka,
                _PpAttr.vm_hv_pu,
                _PpAttr.va_hv_degree,
                _PpAttr.vm_lv_pu,
                _PpAttr.va_lv_degree,
                _PpAttr.loading_percent,
            ],
            index=self._get_pp_ids(_PpTable.trafo, pgm_output_transformers[AttributeType.id]),
        )
        pp_output_trafos[_PpAttr.p_hv_mw] = pgm_output_transformers[AttributeType.p_from] * 1e-6
        pp_output_trafos[_PpAttr.q_hv_mvar] = pgm_output_transformers[AttributeType.q_from] * 1e-6
        pp_output_trafos[_PpAttr.p_lv_mw] = pgm_output_transformers[AttributeType.p_to] * 1e-6
        pp_output_trafos[_PpAttr.q_lv_mvar] = pgm_output_transformers[AttributeType.q_to] * 1e-6
        pp_output_trafos[_PpAttr.pl_mw] = (
            pgm_output_transformers[AttributeType.p_from] + pgm_output_transformers[AttributeType.p_to]
        ) * 1e-6
        pp_output_trafos[_PpAttr.ql_mvar] = (
            pgm_output_transformers[AttributeType.q_from] + pgm_output_transformers[AttributeType.q_to]
        ) * 1e-6
        pp_output_trafos[_PpAttr.i_hv_ka] = pgm_output_transformers[AttributeType.i_from] * 1e-3
        pp_output_trafos[_PpAttr.i_lv_ka] = pgm_output_transformers[AttributeType.i_to] * 1e-3
        pp_output_trafos[_PpAttr.vm_hv_pu] = from_nodes["u_pu"].values
        pp_output_trafos[_PpAttr.vm_lv_pu] = to_nodes["u_pu"].values
        pp_output_trafos[_PpAttr.va_hv_degree] = from_nodes["u_degree"].values
        pp_output_trafos[_PpAttr.va_lv_degree] = to_nodes["u_degree"].values
        pp_output_trafos[_PpAttr.loading_percent] = loading

        self.pp_output_data[_PpTable.res_trafo] = pp_output_trafos

    def _pp_trafos3w_output(self):
        """
        This function converts a power-grid-model Three Winding Transformer output array to a Three Winding Transformer
        Dataframe of PandaPower.

        Returns:
            a PandaPower Dataframe for the Three Winding Transformer component
        """
        if _PpTable.res_trafo3w in self.pp_output_data:
            raise ValueError("res_trafo3w already exists in pp_output_data.")

        if (
            ComponentType.three_winding_transformer not in self.pgm_output_data
            or self.pgm_output_data[ComponentType.three_winding_transformer].size == 0
        ):
            return

        pgm_input_transformers3w = self.pgm_input_data[ComponentType.three_winding_transformer]

        pgm_output_transformers3w = self.pgm_output_data[ComponentType.three_winding_transformer]

        nodes_1 = self.pgm_nodes_lookup.loc[pgm_input_transformers3w[AttributeType.node_1]]
        nodes_2 = self.pgm_nodes_lookup.loc[pgm_input_transformers3w[AttributeType.node_2]]
        nodes_3 = self.pgm_nodes_lookup.loc[pgm_input_transformers3w[AttributeType.node_3]]

        pp_output_trafos3w = pd.DataFrame(
            columns=[
                _PpAttr.p_hv_mw,
                _PpAttr.q_hv_mvar,
                _PpAttr.p_mv_mw,
                _PpAttr.q_mv_mvar,
                _PpAttr.p_lv_mw,
                _PpAttr.q_lv_mvar,
                _PpAttr.pl_mw,
                _PpAttr.ql_mvar,
                _PpAttr.i_hv_ka,
                _PpAttr.i_mv_ka,
                _PpAttr.i_lv_ka,
                _PpAttr.vm_hv_pu,
                _PpAttr.vm_mv_pu,
                _PpAttr.vm_lv_pu,
                _PpAttr.va_hv_degree,
                _PpAttr.va_mv_degree,
                _PpAttr.va_lv_degree,
                _PpAttr.loading_percent,
            ],
            index=self._get_pp_ids(_PpTable.trafo3w, pgm_output_transformers3w[AttributeType.id]),
        )

        pp_output_trafos3w[_PpAttr.p_hv_mw] = pgm_output_transformers3w[AttributeType.p_1] * 1e-6
        pp_output_trafos3w[_PpAttr.q_hv_mvar] = pgm_output_transformers3w[AttributeType.q_1] * 1e-6
        pp_output_trafos3w[_PpAttr.p_mv_mw] = pgm_output_transformers3w[AttributeType.p_2] * 1e-6
        pp_output_trafos3w[_PpAttr.q_mv_mvar] = pgm_output_transformers3w[AttributeType.q_2] * 1e-6
        pp_output_trafos3w[_PpAttr.p_lv_mw] = pgm_output_transformers3w[AttributeType.p_3] * 1e-6
        pp_output_trafos3w[_PpAttr.q_lv_mvar] = pgm_output_transformers3w[AttributeType.q_3] * 1e-6
        pp_output_trafos3w[_PpAttr.pl_mw] = (
            pgm_output_transformers3w[AttributeType.p_1]
            + pgm_output_transformers3w[AttributeType.p_2]
            + pgm_output_transformers3w[AttributeType.p_3]
        ) * 1e-6
        pp_output_trafos3w[_PpAttr.ql_mvar] = (
            pgm_output_transformers3w[AttributeType.q_1]
            + pgm_output_transformers3w[AttributeType.q_2]
            + pgm_output_transformers3w[AttributeType.q_3]
        ) * 1e-6
        pp_output_trafos3w[_PpAttr.i_hv_ka] = pgm_output_transformers3w[AttributeType.i_1] * 1e-3
        pp_output_trafos3w[_PpAttr.i_mv_ka] = pgm_output_transformers3w[AttributeType.i_2] * 1e-3
        pp_output_trafos3w[_PpAttr.i_lv_ka] = pgm_output_transformers3w[AttributeType.i_3] * 1e-3
        pp_output_trafos3w[_PpAttr.vm_hv_pu] = nodes_1["u_pu"].values
        pp_output_trafos3w[_PpAttr.vm_mv_pu] = nodes_2["u_pu"].values
        pp_output_trafos3w[_PpAttr.vm_lv_pu] = nodes_3["u_pu"].values
        pp_output_trafos3w[_PpAttr.va_hv_degree] = nodes_1["u_degree"].values
        pp_output_trafos3w[_PpAttr.va_mv_degree] = nodes_2["u_degree"].values
        pp_output_trafos3w[_PpAttr.va_lv_degree] = nodes_3["u_degree"].values
        pp_output_trafos3w[_PpAttr.loading_percent] = pgm_output_transformers3w[AttributeType.loading] * 1e2

        self.pp_output_data[_PpTable.res_trafo3w] = pp_output_trafos3w

    def _pp_asym_loads_output(self):
        """
        This function converts a power-grid-model Asymmetrical Load output array to an Asymmetrical Load Dataframe of
        PandaPower.

        Returns:
            a PandaPower Dataframe for the Asymmetrical Load component
        """
        if _PpTable.res_asymmetric_load in self.pp_output_data:
            raise ValueError("res_asymmetric_load already exists in pp_output_data.")

        if (
            ComponentType.asym_load not in self.pgm_output_data
            or self.pgm_output_data[ComponentType.asym_load].size == 0
        ):
            return

        pgm_output_asym_loads = self.pgm_output_data[ComponentType.asym_load]

        pp_asym_output_loads = pd.DataFrame(
            columns=[_PpAttr.p_mw, _PpAttr.q_mvar],
            index=self._get_pp_ids(_PpTable.asymmetric_load, pgm_output_asym_loads[AttributeType.id]),
        )

        pp_asym_output_loads[_PpAttr.p_mw] = pgm_output_asym_loads[AttributeType.p] * 1e-6
        pp_asym_output_loads[_PpAttr.q_mvar] = pgm_output_asym_loads[AttributeType.q] * 1e-6

        self.pp_output_data[_PpTable.res_asymmetric_load] = pp_asym_output_loads

    def _pp_asym_gens_output(self):
        """
        This function converts a power-grid-model Asymmetrical Generator output array to an Asymmetric Static Generator
        Dataframe of PandaPower.

        Returns:
            a PandaPower Dataframe for the Asymmetric Static Generator component
        """
        if _PpTable.res_asymmetric_sgen in self.pp_output_data:
            raise ValueError("res_asymmetric_sgen already exists in pp_output_data.")

        if ComponentType.asym_gen not in self.pgm_output_data or self.pgm_output_data[ComponentType.asym_gen].size == 0:
            return

        pgm_output_asym_gens = self.pgm_output_data[ComponentType.asym_gen]

        pp_output_asym_gens = pd.DataFrame(
            columns=[_PpAttr.p_mw, _PpAttr.q_mvar],
            index=self._get_pp_ids(_PpTable.asymmetric_sgen, pgm_output_asym_gens[AttributeType.id]),
        )

        pp_output_asym_gens[_PpAttr.p_mw] = pgm_output_asym_gens[AttributeType.p] * 1e-6
        pp_output_asym_gens[_PpAttr.q_mvar] = pgm_output_asym_gens[AttributeType.q] * 1e-6

        self.pp_output_data[_PpTable.res_asymmetric_sgen] = pp_output_asym_gens

    def _pp_load_elements_output(self, element, symmetric):
        """
        Utility function to convert output of elements represented as load
        in power grid model.
        element: _PpTable.load, _PpTable.motor or _PpTable.ward
        symmetric: True or False
        """
        res_table = "res_" + element if symmetric else "res_" + element + "_3ph"
        if res_table in self.pp_output_data:
            raise ValueError(f"{res_table} already exists in pp_output_data.")

        if element == _PpTable.load:
            load_id_names = ["const_power", "const_impedance", "const_current"]
        elif element == _PpTable.ward:
            load_id_names = ["ward_const_power_load", "ward_const_impedance_load"]
        elif element == _PpTable.motor:
            load_id_names = ["motor_load"]

        if (
            ComponentType.sym_load not in self.pgm_output_data
            or self.pgm_output_data[ComponentType.sym_load].size == 0
            or (element, load_id_names[0]) not in self.idx
        ):
            return
        # Store the results
        self.pp_output_data[res_table] = self._pp_load_result_accumulate(
            pp_component_name=element, load_id_names=load_id_names
        )

    def _pp_load_result_accumulate(self, pp_component_name: str, load_id_names: list[str]) -> pd.DataFrame:
        """
        This function converts a power-grid-model Symmetrical and asymmetrical load output array
        to a respective Dataframe of PandaPower.

        Returns:
            a PandaPower Dataframe for the Load component
        """
        # Create a DataFrame wih all the pgm output loads and index it in the pgm id
        pgm_output_loads = self.pgm_output_data[ComponentType.sym_load]
        # Sum along rows for asym output
        active_power = (
            pgm_output_loads[AttributeType.p].sum(axis=1)
            if pgm_output_loads[AttributeType.p].ndim == ARRAY_2D
            else pgm_output_loads[AttributeType.p]
        )
        reactive_power = (
            pgm_output_loads[AttributeType.q].sum(axis=1)
            if pgm_output_loads[AttributeType.q].ndim == ARRAY_2D
            else pgm_output_loads[AttributeType.q]
        )
        all_loads = pd.DataFrame({"p": active_power, "q": reactive_power}, index=pgm_output_loads[AttributeType.id])

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
            accumulated_loads = accumulated_loads.add(selected_loads, fill_value=0.0)

        # Multiply the values and rename the columns to match pandapower
        accumulated_loads *= 1e-6
        accumulated_loads.columns = [_PpAttr.p_mw, _PpAttr.q_mvar]

        return accumulated_loads

    def _pp_switches_output(self):
        """
        This function converts a power-grid-model links, lines, transformers, transformers3w output array
        to res_switch Dataframe of PandaPower.
        Switch results are only possible at round conversions. ie, input switch data is available
        """
        switch_data_unavailable = _PpTable.switch not in self.pp_input_data
        links_absent = (
            ComponentType.link not in self.pgm_output_data or self.pgm_output_data[ComponentType.link].size == 0
        )
        rest_switches_absent = {
            pp_comp: ("res_" + pp_comp not in self.pp_output_data)
            for pp_comp in [_PpTable.line, _PpTable.trafo, _PpTable.trafo3w]
        }
        if (all(rest_switches_absent.values()) and links_absent) or switch_data_unavailable:
            return

        if _PpTable.res_switch in self.pp_output_data:
            raise ValueError("res_switch already exists in pp_output_data")

        def join_currents(table: str, bus_name: str, i_name: str) -> pd.DataFrame:
            # Create a dataframe of element: input table index, bus: input branch bus, current: output current
            single_df = self.pp_input_data[table][[bus_name]]
            single_df = single_df.join(self.pp_output_data["res_" + table][i_name])
            single_df.columns = [_PpTable.bus, _PpAttr.i_ka]
            single_df[_PpAttr.element] = single_df.index
            single_df[_PpAttr.et] = table_to_et[table]
            return single_df

        switch_attrs = {
            _PpTable.trafo: {_PpAttr.hv_bus: _PpAttr.i_hv_ka, _PpAttr.lv_bus: _PpAttr.i_lv_ka},
            _PpTable.trafo3w: {
                _PpAttr.hv_bus: _PpAttr.i_hv_ka,
                _PpAttr.mv_bus: _PpAttr.i_mv_ka,
                _PpAttr.lv_bus: _PpAttr.i_lv_ka,
            },
            _PpTable.line: {_PpAttr.from_bus: _PpAttr.i_from_ka, _PpAttr.to_bus: _PpAttr.i_to_ka},
        }
        table_to_et = {_PpTable.trafo: "t", _PpTable.trafo3w: "t3", _PpTable.line: "l"}

        # Prepare output dataframe, save index for later
        pp_switches_output = self.pp_input_data[_PpTable.switch]
        pp_switches_output_index = pp_switches_output.index

        # Combine all branch bus, current and et in one dataframe
        dfs = [
            join_currents(table, bus_name, i_name)
            for table, attr_names in switch_attrs.items()
            for bus_name, i_name in attr_names.items()
            if not rest_switches_absent[table]
        ]
        all_i_df = (
            pd.DataFrame(columns=[_PpTable.bus, _PpAttr.element, _PpAttr.et, _PpAttr.i_ka])
            if not dfs
            else pd.concat(dfs)
        )

        # Merge on input data to get current and drop other columns
        pp_switches_output = pd.merge(
            pp_switches_output,
            all_i_df,
            how="left",
            left_on=[_PpTable.bus, _PpAttr.element, _PpAttr.et],
            right_on=[_PpTable.bus, _PpAttr.element, _PpAttr.et],
        )
        pp_switches_output = pp_switches_output[[_PpAttr.i_ka]]
        pp_switches_output.set_index(pp_switches_output_index, inplace=True)

        # For et=b, ie bus to bus switches, links are created. get result from them
        if not links_absent:
            links = self.pgm_output_data[ComponentType.link]
            # For links, i_from = i_to = i_ka / 1e3
            link_ids = self._get_pp_ids(_PpTable.switch, links["id"], "b2b_switches")
            pp_switches_output.loc[link_ids, _PpAttr.i_ka] = links["i_from"] * 1e-3
        in_ka = self.pp_input_data[_PpTable.switch][_PpAttr.in_ka].values
        pp_switches_output[_PpAttr.loading_percent] = np.nan
        pp_switches_output[_PpAttr.loading_percent] = np.divide(
            pp_switches_output[_PpAttr.i_ka].values, in_ka, where=in_ka != 0, out=None
        )

        self.pp_output_data[_PpTable.res_switch] = pp_switches_output

    def _pp_buses_output_3ph(self):
        """
        This function converts a power-grid-model Node output array to a Bus Dataframe of PandaPower.

        Returns:
            a PandaPower Dataframe for the Bus component
        """
        if _PpTable.res_bus_3ph in self.pp_output_data:
            raise ValueError("res_bus_3ph already exists in pp_output_data.")

        if ComponentType.node not in self.pgm_output_data or self.pgm_output_data[ComponentType.node].size == 0:
            return

        pgm_nodes = self.pgm_output_data[ComponentType.node]

        pp_output_buses_3ph = pd.DataFrame(
            columns=[
                _PpAttr.vm_a_pu,
                _PpAttr.va_a_degree,
                _PpAttr.vm_b_pu,
                _PpAttr.va_b_degree,
                _PpAttr.vm_c_pu,
                _PpAttr.va_c_degree,
                _PpAttr.p_a_mw,
                _PpAttr.q_a_mvar,
                _PpAttr.p_b_mw,
                _PpAttr.q_b_mvar,
                _PpAttr.p_c_mw,
                _PpAttr.q_c_mvar,
                _PpAttr.unbalance_percent,
            ],
            index=self._get_pp_ids(_PpTable.bus, pgm_nodes[AttributeType.id]),
        )

        node_u_pu = pgm_nodes[AttributeType.u_pu]
        node_u_angle = pgm_nodes[AttributeType.u_angle]
        u_pu_and_angle = node_u_pu * np.exp(1j * node_u_angle)
        # Phase to sequence transformation U_012.T = 1/3 . Transformation_matrix x U_abc.T
        alpha = np.exp(1j * np.pi * 120 / 180)
        trans_matrix = np.array([[1, 1, 1], [1, alpha, alpha * alpha], [1, alpha * alpha, alpha]])
        u_sequence = (1 / 3) * np.matmul(trans_matrix, u_pu_and_angle.T).T

        pp_output_buses_3ph[_PpAttr.vm_a_pu] = node_u_pu[:, 0]
        pp_output_buses_3ph[_PpAttr.va_a_degree] = node_u_angle[:, 0] * (180.0 / np.pi)
        pp_output_buses_3ph[_PpAttr.vm_b_pu] = node_u_pu[:, 1]
        pp_output_buses_3ph[_PpAttr.va_b_degree] = node_u_angle[:, 1] * (180.0 / np.pi)
        pp_output_buses_3ph[_PpAttr.vm_c_pu] = node_u_pu[:, 2]
        pp_output_buses_3ph[_PpAttr.va_c_degree] = node_u_angle[:, 2] * (180.0 / np.pi)
        pp_output_buses_3ph[_PpAttr.unbalance_percent] = np.abs(u_sequence[:, 2]) / np.abs(u_sequence[:, 1]) * 100

        # p_to, p_from, q_to and q_from connected to the bus have to be summed up
        self._pp_buses_output_3ph__accumulate_power(pp_output_buses_3ph)

        self.pp_output_data[_PpTable.res_bus_3ph] = pp_output_buses_3ph

    def _pp_buses_output_3ph__accumulate_power(self, pp_output_buses_3ph: pd.DataFrame):
        """
        For each node, we accumulate the power for all connected branches and branches for asymmetric output

        Args:
            pp_output_buses: a Pandapower output dataframe of Bus component

        Returns:
            accumulated power for each bus
        """
        power_columns = [
            _PpAttr.p_a_mw,
            _PpAttr.p_b_mw,
            _PpAttr.p_c_mw,
            _PpAttr.q_a_mvar,
            _PpAttr.q_b_mvar,
            _PpAttr.q_c_mvar,
        ]
        # Let's define all the components and sides where nodes can be connected
        component_sides = {
            ComponentType.line: [
                (AttributeType.from_node, AttributeType.p_from, AttributeType.q_from),
                (AttributeType.to_node, AttributeType.p_to, AttributeType.q_to),
            ],
            ComponentType.link: [
                (AttributeType.from_node, AttributeType.p_from, AttributeType.q_from),
                (AttributeType.to_node, AttributeType.p_to, AttributeType.q_to),
            ],
            ComponentType.transformer: [
                (AttributeType.from_node, AttributeType.p_from, AttributeType.q_from),
                (AttributeType.to_node, AttributeType.p_to, AttributeType.q_to),
            ],
        }

        # Set the initial powers to zero
        pp_output_buses_3ph[power_columns] = 0.0

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
                        self.pgm_output_data[component][p_col][:, 0],
                        self.pgm_output_data[component][p_col][:, 1],
                        self.pgm_output_data[component][p_col][:, 2],
                        self.pgm_output_data[component][q_col][:, 0],
                        self.pgm_output_data[component][q_col][:, 1],
                        self.pgm_output_data[component][q_col][:, 2],
                    ),
                    columns=[node_col, *power_columns],
                )

                # Accumulate the powers and index by panda power bus index
                accumulated_data = component_data.groupby(node_col).sum()
                accumulated_data.index = self._get_pp_ids(_PpTable.bus, pd.Series(accumulated_data.index))

                # We might not have power data for each pp bus, so select only the indexes for which data is available
                idx = pp_output_buses_3ph.index.intersection(accumulated_data.index)

                # Now add the active and reactive powers to the pp busses
                # Note that the units are incorrect; for efficiency, unit conversions will be applied at the end.
                pp_output_buses_3ph.loc[idx, power_columns] -= accumulated_data[power_columns]

        # Finally apply the unit conversion (W -> MW and VAR -> MVAR)
        pp_output_buses_3ph[power_columns] /= 1e6

    def _pp_lines_output_3ph(self):
        """
        This function converts a power-grid-model Line output array to a Line Dataframe of PandaPower.

        Returns:
            a PandaPower Dataframe for the Line component
        """
        if _PpTable.res_line_3ph in self.pp_output_data:
            raise ValueError("res_line_3ph already exists in pp_output_data.")

        if any(
            (comp not in self.pgm_output_data or self.pgm_output_data[comp].size == 0)
            for comp in [ComponentType.node, ComponentType.line]
        ):
            return

        pgm_input_lines = self.pgm_input_data[ComponentType.line]
        pgm_output_lines = self.pgm_output_data[ComponentType.line]
        pgm_output_nodes = self.pgm_output_data[ComponentType.node]

        u_complex = pd.DataFrame(
            pgm_output_nodes[AttributeType.u] * np.exp(1j * pgm_output_nodes[AttributeType.u_angle]),
            index=self.pgm_output_data[ComponentType.node][AttributeType.id],
        )
        from_nodes = pgm_input_lines[AttributeType.from_node]
        to_nodes = pgm_input_lines[AttributeType.to_node]
        i_from = (
            pgm_output_lines[AttributeType.p_from] + 1j * pgm_output_lines[AttributeType.q_from]
        ) / u_complex.iloc[from_nodes, :]
        i_to = (pgm_output_lines[AttributeType.p_to] + 1j * pgm_output_lines[AttributeType.q_to]) / u_complex.iloc[
            to_nodes, :
        ]

        loss_params = get_loss_params_3ph()
        pp_output_lines_3ph = pd.DataFrame(
            columns=[
                _PpAttr.p_a_from_mw,
                _PpAttr.q_a_from_mvar,
                _PpAttr.p_b_from_mw,
                _PpAttr.q_b_from_mvar,
                _PpAttr.p_c_from_mw,
                _PpAttr.q_c_from_mvar,
                _PpAttr.p_a_to_mw,
                _PpAttr.q_a_to_mvar,
                _PpAttr.p_b_to_mw,
                _PpAttr.q_b_to_mvar,
                _PpAttr.p_c_to_mw,
                _PpAttr.q_c_to_mvar,
                loss_params[0],
                loss_params[1],
                loss_params[2],
                loss_params[3],
                loss_params[4],
                loss_params[5],
                _PpAttr.i_a_from_ka,
                _PpAttr.i_b_from_ka,
                _PpAttr.i_c_from_ka,
                _PpAttr.i_n_from_ka,
                _PpAttr.i_a_ka,
                _PpAttr.i_b_ka,
                _PpAttr.i_c_ka,
                _PpAttr.i_n_ka,
                _PpAttr.i_a_to_ka,
                _PpAttr.i_b_to_ka,
                _PpAttr.i_c_to_ka,
                _PpAttr.i_n_to_ka,
                _PpAttr.loading_a_percent,
                _PpAttr.loading_b_percent,
                _PpAttr.loading_c_percent,
                _PpAttr.loading_percent,
            ],
            index=self._get_pp_ids(_PpTable.line, pgm_output_lines[AttributeType.id]),
        )

        pp_output_lines_3ph[_PpAttr.p_a_from_mw] = pgm_output_lines[AttributeType.p_from][:, 0] * 1e-6
        pp_output_lines_3ph[_PpAttr.q_a_from_mvar] = pgm_output_lines[AttributeType.q_from][:, 0] * 1e-6
        pp_output_lines_3ph[_PpAttr.p_b_from_mw] = pgm_output_lines[AttributeType.p_from][:, 1] * 1e-6
        pp_output_lines_3ph[_PpAttr.q_b_from_mvar] = pgm_output_lines[AttributeType.q_from][:, 1] * 1e-6
        pp_output_lines_3ph[_PpAttr.p_c_from_mw] = pgm_output_lines[AttributeType.p_from][:, 2] * 1e-6
        pp_output_lines_3ph[_PpAttr.q_c_from_mvar] = pgm_output_lines[AttributeType.q_from][:, 2] * 1e-6
        pp_output_lines_3ph[_PpAttr.p_a_to_mw] = pgm_output_lines[AttributeType.p_to][:, 0] * 1e-6
        pp_output_lines_3ph[_PpAttr.q_a_to_mvar] = pgm_output_lines[AttributeType.q_to][:, 0] * 1e-6
        pp_output_lines_3ph[_PpAttr.p_b_to_mw] = pgm_output_lines[AttributeType.p_to][:, 1] * 1e-6
        pp_output_lines_3ph[_PpAttr.q_b_to_mvar] = pgm_output_lines[AttributeType.q_to][:, 1] * 1e-6
        pp_output_lines_3ph[_PpAttr.p_c_to_mw] = pgm_output_lines[AttributeType.p_to][:, 2] * 1e-6
        pp_output_lines_3ph[_PpAttr.q_c_to_mvar] = pgm_output_lines[AttributeType.q_to][:, 2] * 1e-6
        pp_output_lines_3ph[loss_params[0]] = (
            pgm_output_lines[AttributeType.p_from][:, 0] + pgm_output_lines[AttributeType.p_to][:, 0]
        ) * 1e-6
        pp_output_lines_3ph[loss_params[1]] = (
            pgm_output_lines[AttributeType.q_from][:, 0] + pgm_output_lines[AttributeType.q_to][:, 0]
        ) * 1e-6
        pp_output_lines_3ph[loss_params[2]] = (
            pgm_output_lines[AttributeType.p_from][:, 1] + pgm_output_lines[AttributeType.p_to][:, 1]
        ) * 1e-6
        pp_output_lines_3ph[loss_params[3]] = (
            pgm_output_lines[AttributeType.q_from][:, 1] + pgm_output_lines[AttributeType.q_to][:, 1]
        ) * 1e-6
        pp_output_lines_3ph[loss_params[4]] = (
            pgm_output_lines[AttributeType.p_from][:, 2] + pgm_output_lines[AttributeType.p_to][:, 2]
        ) * 1e-6
        pp_output_lines_3ph[loss_params[5]] = (
            pgm_output_lines[AttributeType.q_from][:, 2] + pgm_output_lines[AttributeType.q_to][:, 2]
        ) * 1e-6
        pp_output_lines_3ph[_PpAttr.i_a_from_ka] = pgm_output_lines[AttributeType.i_from][:, 0] * 1e-3
        pp_output_lines_3ph[_PpAttr.i_b_from_ka] = pgm_output_lines[AttributeType.i_from][:, 1] * 1e-3
        pp_output_lines_3ph[_PpAttr.i_c_from_ka] = pgm_output_lines[AttributeType.i_from][:, 2] * 1e-3
        pp_output_lines_3ph[_PpAttr.i_n_from_ka] = np.array(np.abs(np.sum(i_from, axis=1))) * 1e-3
        pp_output_lines_3ph[_PpAttr.i_a_to_ka] = pgm_output_lines[AttributeType.i_to][:, 0] * 1e-3
        pp_output_lines_3ph[_PpAttr.i_b_to_ka] = pgm_output_lines[AttributeType.i_to][:, 1] * 1e-3
        pp_output_lines_3ph[_PpAttr.i_c_to_ka] = pgm_output_lines[AttributeType.i_to][:, 2] * 1e-3
        pp_output_lines_3ph[_PpAttr.i_n_to_ka] = np.array(np.abs(np.sum(i_to, axis=1))) * 1e-3
        pp_output_lines_3ph[_PpAttr.i_a_ka] = np.maximum(
            pp_output_lines_3ph[_PpAttr.i_a_from_ka], pp_output_lines_3ph[_PpAttr.i_a_to_ka]
        )
        pp_output_lines_3ph[_PpAttr.i_b_ka] = np.maximum(
            pp_output_lines_3ph[_PpAttr.i_b_from_ka], pp_output_lines_3ph[_PpAttr.i_b_to_ka]
        )
        pp_output_lines_3ph[_PpAttr.i_c_ka] = np.maximum(
            pp_output_lines_3ph[_PpAttr.i_c_from_ka], pp_output_lines_3ph[_PpAttr.i_c_to_ka]
        )
        pp_output_lines_3ph[_PpAttr.i_n_ka] = np.maximum(
            pp_output_lines_3ph[_PpAttr.i_n_from_ka], pp_output_lines_3ph[_PpAttr.i_n_to_ka]
        )
        pp_output_lines_3ph[_PpAttr.loading_a_percent] = (
            np.maximum(pp_output_lines_3ph[_PpAttr.i_a_from_ka], pp_output_lines_3ph[_PpAttr.i_a_to_ka])
            / pgm_input_lines["i_n"]
        ) * 1e5
        pp_output_lines_3ph[_PpAttr.loading_b_percent] = (
            np.maximum(pp_output_lines_3ph[_PpAttr.i_b_from_ka], pp_output_lines_3ph[_PpAttr.i_b_to_ka])
            / pgm_input_lines["i_n"]
        ) * 1e5
        pp_output_lines_3ph[_PpAttr.loading_c_percent] = (
            np.maximum(pp_output_lines_3ph[_PpAttr.i_c_from_ka], pp_output_lines_3ph[_PpAttr.i_c_to_ka])
            / pgm_input_lines["i_n"]
        ) * 1e5
        pp_output_lines_3ph[_PpAttr.loading_percent] = np.maximum(
            np.maximum(pp_output_lines_3ph[_PpAttr.loading_a_percent], pp_output_lines_3ph[_PpAttr.loading_b_percent]),
            pp_output_lines_3ph[_PpAttr.loading_c_percent],
        )

        self.pp_output_data[_PpTable.res_line_3ph] = pp_output_lines_3ph

    def _pp_ext_grids_output_3ph(self):
        """
        This function converts a power-grid-model Source output array to an External Grid Dataframe of PandaPower.

        Returns:
            a PandaPower Dataframe for the External Grid component
        """
        if _PpTable.res_ext_grid_3ph in self.pp_output_data:
            raise ValueError("res_ext_grid_3ph already exists in pp_output_data.")

        if ComponentType.source not in self.pgm_output_data or self.pgm_output_data[ComponentType.source].size == 0:
            return

        pgm_output_sources = self.pgm_output_data[ComponentType.source]

        pp_output_ext_grids_3ph = pd.DataFrame(
            columns=[
                _PpAttr.p_a_mw,
                _PpAttr.q_a_mvar,
                _PpAttr.p_b_mw,
                _PpAttr.q_b_mvar,
                _PpAttr.p_c_mw,
                _PpAttr.q_c_mvar,
            ],
            index=self._get_pp_ids(_PpTable.ext_grid, pgm_output_sources[AttributeType.id]),
        )
        pp_output_ext_grids_3ph[_PpAttr.p_a_mw] = pgm_output_sources[AttributeType.p][:, 0] * 1e-6
        pp_output_ext_grids_3ph[_PpAttr.q_a_mvar] = pgm_output_sources[AttributeType.q][:, 0] * 1e-6
        pp_output_ext_grids_3ph[_PpAttr.p_b_mw] = pgm_output_sources[AttributeType.p][:, 1] * 1e-6
        pp_output_ext_grids_3ph[_PpAttr.q_b_mvar] = pgm_output_sources[AttributeType.q][:, 1] * 1e-6
        pp_output_ext_grids_3ph[_PpAttr.p_c_mw] = pgm_output_sources[AttributeType.p][:, 2] * 1e-6
        pp_output_ext_grids_3ph[_PpAttr.q_c_mvar] = pgm_output_sources[AttributeType.q][:, 2] * 1e-6

        self.pp_output_data[_PpTable.res_ext_grid_3ph] = pp_output_ext_grids_3ph

    def _pp_sgens_output_3ph(self):
        """
        This function converts a power-grid-model Symmetrical Generator output array to a Static Generator Dataframe of
        PandaPower.

        Returns:
            a PandaPower Dataframe for the Static Generator component
        """
        if _PpTable.res_sgen_3ph in self.pp_output_data:
            raise ValueError("res_sgen_3ph already exists in pp_output_data.")

        if ComponentType.sym_gen not in self.pgm_output_data or self.pgm_output_data[ComponentType.sym_gen].size == 0:
            return

        pgm_output_sym_gens = self.pgm_output_data[ComponentType.sym_gen]

        pp_output_sgens = pd.DataFrame(
            columns=[_PpAttr.p_mw, _PpAttr.q_mvar],
            index=self._get_pp_ids(_PpTable.sgen, pgm_output_sym_gens[AttributeType.id]),
        )
        pp_output_sgens[_PpAttr.p_mw] = np.sum(pgm_output_sym_gens[AttributeType.p], axis=1) * 1e-6
        pp_output_sgens[_PpAttr.q_mvar] = np.sum(pgm_output_sym_gens[AttributeType.q], axis=1) * 1e-6

        self.pp_output_data[_PpTable.res_sgen_3ph] = pp_output_sgens

    def _pp_trafos_output_3ph(self):  # noqa: PLR0915  # pylint: disable=too-many-statements
        """
        This function converts a power-grid-model Transformer output array to a Transformer Dataframe of
        PandaPower.

        Returns:
            a PandaPower Dataframe for the Transformer component
        """
        if _PpTable.res_trafo_3ph in self.pp_output_data:
            raise ValueError("res_trafo_3ph already exists in pp_output_data.")

        if (
            ComponentType.transformer not in self.pgm_output_data
            or self.pgm_output_data[ComponentType.transformer].size == 0
        ) or (_PpTable.trafo not in self.pp_input_data or len(self.pp_input_data[_PpTable.trafo]) == 0):
            return

        pgm_input_transformers = self.pgm_input_data[ComponentType.transformer]
        pp_input_transformers = self.pp_input_data[_PpTable.trafo]
        pgm_output_transformers = self.pgm_output_data[ComponentType.transformer]

        # Only derating factor used here. Sn is already being multiplied by parallel
        loading_multiplier = pp_input_transformers["df"] * 1e2
        if self.trafo_loading == "current":
            ui_from = pgm_output_transformers[AttributeType.i_from] * pgm_input_transformers[AttributeType.u1][:, None]
            ui_to = pgm_output_transformers[AttributeType.i_to] * pgm_input_transformers[AttributeType.u2][:, None]
            loading_a_percent = (
                np.sqrt(3) * np.maximum(ui_from[:, 0], ui_to[:, 0]) / pgm_input_transformers[AttributeType.sn]
            )
            loading_b_percent = (
                np.sqrt(3) * np.maximum(ui_from[:, 1], ui_to[:, 1]) / pgm_input_transformers[AttributeType.sn]
            )
            loading_c_percent = (
                np.sqrt(3) * np.maximum(ui_from[:, 2], ui_to[:, 2]) / pgm_input_transformers[AttributeType.sn]
            )
        elif self.trafo_loading == "power":
            loading_a_percent = np.maximum(
                pgm_output_transformers[AttributeType.s_from][:, 0],
                pgm_output_transformers[AttributeType.s_to][:, 0],
            ) / (pgm_input_transformers[AttributeType.sn] / 3)
            loading_b_percent = np.maximum(
                pgm_output_transformers[AttributeType.s_from][:, 1],
                pgm_output_transformers[AttributeType.s_to][:, 1],
            ) / (pgm_input_transformers[AttributeType.sn] / 3)
            loading_c_percent = np.maximum(
                pgm_output_transformers[AttributeType.s_from][:, 2],
                pgm_output_transformers[AttributeType.s_to][:, 2],
            ) / (pgm_input_transformers[AttributeType.sn] / 3)
        else:
            raise ValueError(f"Invalid transformer loading type: {self.trafo_loading!s}")

        loading = np.maximum(np.maximum(loading_a_percent, loading_b_percent), loading_c_percent)

        loss_params = get_loss_params_3ph()
        pp_output_trafos_3ph = pd.DataFrame(
            columns=[
                _PpAttr.p_a_hv_mw,
                _PpAttr.q_a_hv_mvar,
                _PpAttr.p_b_hv_mw,
                _PpAttr.q_b_hv_mvar,
                _PpAttr.p_c_hv_mw,
                _PpAttr.q_c_hv_mvar,
                _PpAttr.p_a_lv_mw,
                _PpAttr.q_a_lv_mvar,
                _PpAttr.p_b_lv_mw,
                _PpAttr.q_b_lv_mvar,
                _PpAttr.p_c_lv_mw,
                _PpAttr.q_c_lv_mvar,
                loss_params[0],
                loss_params[1],
                loss_params[2],
                loss_params[3],
                loss_params[4],
                loss_params[5],
                _PpAttr.i_a_hv_ka,
                _PpAttr.i_a_lv_ka,
                _PpAttr.i_b_hv_ka,
                _PpAttr.i_b_lv_ka,
                _PpAttr.i_c_hv_ka,
                _PpAttr.i_c_lv_ka,
                _PpAttr.loading_a_percent,
                _PpAttr.loading_b_percent,
                _PpAttr.loading_c_percent,
                _PpAttr.loading_percent,
            ],
            index=self._get_pp_ids(_PpTable.trafo, pgm_output_transformers[AttributeType.id]),
        )
        pp_output_trafos_3ph[_PpAttr.p_a_hv_mw] = pgm_output_transformers[AttributeType.p_from][:, 0] * 1e-6
        pp_output_trafos_3ph[_PpAttr.q_a_hv_mvar] = pgm_output_transformers[AttributeType.q_from][:, 0] * 1e-6
        pp_output_trafos_3ph[_PpAttr.p_b_hv_mw] = pgm_output_transformers[AttributeType.p_from][:, 1] * 1e-6
        pp_output_trafos_3ph[_PpAttr.q_b_hv_mvar] = pgm_output_transformers[AttributeType.q_from][:, 1] * 1e-6
        pp_output_trafos_3ph[_PpAttr.p_c_hv_mw] = pgm_output_transformers[AttributeType.p_from][:, 2] * 1e-6
        pp_output_trafos_3ph[_PpAttr.q_c_hv_mvar] = pgm_output_transformers[AttributeType.q_from][:, 2] * 1e-6
        pp_output_trafos_3ph[_PpAttr.p_a_lv_mw] = pgm_output_transformers[AttributeType.p_to][:, 0] * 1e-6
        pp_output_trafos_3ph[_PpAttr.q_a_lv_mvar] = pgm_output_transformers[AttributeType.q_to][:, 0] * 1e-6
        pp_output_trafos_3ph[_PpAttr.p_b_lv_mw] = pgm_output_transformers[AttributeType.p_to][:, 1] * 1e-6
        pp_output_trafos_3ph[_PpAttr.q_b_lv_mvar] = pgm_output_transformers[AttributeType.q_to][:, 1] * 1e-6
        pp_output_trafos_3ph[_PpAttr.p_c_lv_mw] = pgm_output_transformers[AttributeType.p_to][:, 2] * 1e-6
        pp_output_trafos_3ph[_PpAttr.q_c_lv_mvar] = pgm_output_transformers[AttributeType.q_to][:, 2] * 1e-6
        pp_output_trafos_3ph[loss_params[0]] = (
            pgm_output_transformers[AttributeType.p_from][:, 0] + pgm_output_transformers[AttributeType.p_to][:, 0]
        ) * 1e-6
        pp_output_trafos_3ph[loss_params[1]] = (
            pgm_output_transformers[AttributeType.q_from][:, 0] + pgm_output_transformers[AttributeType.q_to][:, 0]
        ) * 1e-6
        pp_output_trafos_3ph[loss_params[2]] = (
            pgm_output_transformers[AttributeType.p_from][:, 1] + pgm_output_transformers[AttributeType.p_to][:, 1]
        ) * 1e-6
        pp_output_trafos_3ph[loss_params[3]] = (
            pgm_output_transformers[AttributeType.q_from][:, 1] + pgm_output_transformers[AttributeType.q_to][:, 1]
        ) * 1e-6
        pp_output_trafos_3ph[loss_params[4]] = (
            pgm_output_transformers[AttributeType.p_from][:, 2] + pgm_output_transformers[AttributeType.p_to][:, 2]
        ) * 1e-6
        pp_output_trafos_3ph[loss_params[5]] = (
            pgm_output_transformers[AttributeType.q_from][:, 2] + pgm_output_transformers[AttributeType.q_to][:, 2]
        ) * 1e-6
        pp_output_trafos_3ph[_PpAttr.i_a_hv_ka] = pgm_output_transformers[AttributeType.i_from][:, 0] * 1e-3
        pp_output_trafos_3ph[_PpAttr.i_a_lv_ka] = pgm_output_transformers[AttributeType.i_to][:, 0] * 1e-3
        pp_output_trafos_3ph[_PpAttr.i_b_hv_ka] = pgm_output_transformers[AttributeType.i_from][:, 1] * 1e-3
        pp_output_trafos_3ph[_PpAttr.i_b_lv_ka] = pgm_output_transformers[AttributeType.i_to][:, 1] * 1e-3
        pp_output_trafos_3ph[_PpAttr.i_c_hv_ka] = pgm_output_transformers[AttributeType.i_from][:, 2] * 1e-3
        pp_output_trafos_3ph[_PpAttr.i_c_lv_ka] = pgm_output_transformers[AttributeType.i_to][:, 2] * 1e-3
        pp_output_trafos_3ph[_PpAttr.loading_a_percent] = loading_a_percent * loading_multiplier
        pp_output_trafos_3ph[_PpAttr.loading_b_percent] = loading_b_percent * loading_multiplier
        pp_output_trafos_3ph[_PpAttr.loading_c_percent] = loading_c_percent * loading_multiplier
        pp_output_trafos_3ph[_PpAttr.loading_percent] = loading * loading_multiplier

        self.pp_output_data[_PpTable.res_trafo_3ph] = pp_output_trafos_3ph

    def _pp_shunts_output_3ph(self):
        """
        This function converts a power-grid-model Shunt output array to a Shunt Dataframe of PandaPower.

        Returns:
            a PandaPower Dataframe for the Shunt component
        """
        if _PpTable.res_shunt_3ph in self.pp_output_data:
            raise ValueError("res_shunt_3ph already exists in pp_output_data.")

        if ComponentType.shunt not in self.pgm_output_data or self.pgm_output_data[ComponentType.shunt].size == 0:
            return

        pgm_output_shunts = self.pgm_output_data[ComponentType.shunt]

        pp_output_shunts = pd.DataFrame(
            columns=[_PpAttr.p_mw, _PpAttr.q_mvar, _PpAttr.vm_pu],
            index=self._get_pp_ids(_PpTable.shunt, pgm_output_shunts[AttributeType.id]),
        )
        pp_output_shunts[_PpAttr.p_mw] = pgm_output_shunts[AttributeType.p].sum() * 1e-6
        pp_output_shunts[_PpAttr.q_mvar] = pgm_output_shunts[AttributeType.q].sum() * 1e-6
        # TODO Find a better way for mapping vm_pu from bus
        # pp_output_shunts[_PpAttr.vm_pu] = np.nan
        self.pp_output_data[_PpTable.res_shunt_3ph] = pp_output_shunts

    def _pp_asym_loads_output_3ph(self):
        """
        This function converts a power-grid-model Asymmetrical Load output array to an Asymmetrical Load Dataframe of
        PandaPower.

        Returns:
            a PandaPower Dataframe for the Asymmetrical Load component
        """
        if _PpTable.res_asymmetric_load_3ph in self.pp_output_data:
            raise ValueError("res_asymmetric_load_3ph already exists in pp_output_data.")

        if (
            ComponentType.asym_load not in self.pgm_output_data
            or self.pgm_output_data[ComponentType.asym_load].size == 0
        ):
            return

        pgm_output_asym_loads = self.pgm_output_data[ComponentType.asym_load]

        pp_asym_load_p = pgm_output_asym_loads[AttributeType.p] * 1e-6
        pp_asym_load_q = pgm_output_asym_loads[AttributeType.q] * 1e-6

        pp_asym_output_loads_3ph = pd.DataFrame(
            columns=[
                _PpAttr.p_a_mw,
                _PpAttr.q_a_mvar,
                _PpAttr.p_b_mw,
                _PpAttr.q_b_mvar,
                _PpAttr.p_c_mw,
                _PpAttr.q_c_mvar,
            ],
            index=self._get_pp_ids(_PpTable.asymmetric_load, pgm_output_asym_loads[AttributeType.id]),
        )

        pp_asym_output_loads_3ph[_PpAttr.p_a_mw] = pp_asym_load_p[:, 0]
        pp_asym_output_loads_3ph[_PpAttr.q_a_mvar] = pp_asym_load_q[:, 0]
        pp_asym_output_loads_3ph[_PpAttr.p_b_mw] = pp_asym_load_p[:, 1]
        pp_asym_output_loads_3ph[_PpAttr.q_b_mvar] = pp_asym_load_q[:, 1]
        pp_asym_output_loads_3ph[_PpAttr.p_c_mw] = pp_asym_load_p[:, 2]
        pp_asym_output_loads_3ph[_PpAttr.q_c_mvar] = pp_asym_load_q[:, 2]

        self.pp_output_data[_PpTable.res_asymmetric_load_3ph] = pp_asym_output_loads_3ph

    def _pp_asym_gens_output_3ph(self):
        """
        This function converts a power-grid-model Asymmetrical Generator output array to an Asymmetric Static Generator
        Dataframe of PandaPower.

        Returns:
            a PandaPower Dataframe for the Asymmetric Static Generator component
        """
        if _PpTable.res_asymmetric_sgen_3ph in self.pp_output_data:
            raise ValueError("res_asymmetric_sgen_3ph already exists in pp_output_data.")

        if "asym_gen" not in self.pgm_output_data or self.pgm_output_data[ComponentType.asym_gen].size == 0:
            return

        pgm_output_asym_gens = self.pgm_output_data[ComponentType.asym_gen]

        pp_asym_gen_p = pgm_output_asym_gens[AttributeType.p] * 1e-6
        pp_asym_gen_q = pgm_output_asym_gens[AttributeType.q] * 1e-6

        pp_output_asym_gens_3ph = pd.DataFrame(
            columns=[
                _PpAttr.p_a_mw,
                _PpAttr.q_a_mvar,
                _PpAttr.p_b_mw,
                _PpAttr.q_b_mvar,
                _PpAttr.p_c_mw,
                _PpAttr.q_c_mvar,
            ],
            index=self._get_pp_ids(_PpTable.asymmetric_sgen, pgm_output_asym_gens[AttributeType.id]),
        )

        pp_output_asym_gens_3ph[_PpAttr.p_a_mw] = pp_asym_gen_p[:, 0]
        pp_output_asym_gens_3ph[_PpAttr.q_a_mvar] = pp_asym_gen_q[:, 0]
        pp_output_asym_gens_3ph[_PpAttr.p_b_mw] = pp_asym_gen_p[:, 1]
        pp_output_asym_gens_3ph[_PpAttr.q_b_mvar] = pp_asym_gen_q[:, 1]
        pp_output_asym_gens_3ph[_PpAttr.p_c_mw] = pp_asym_gen_p[:, 2]
        pp_output_asym_gens_3ph[_PpAttr.q_c_mvar] = pp_asym_gen_q[:, 2]

        self.pp_output_data[_PpTable.res_asymmetric_sgen_3ph] = pp_output_asym_gens_3ph

    def _generate_ids(self, pp_table: str, pp_idx: pd.Index, name: str | None = None) -> np.ndarray:
        """
        Generate numerical power-grid-model IDs for a PandaPower component

        Args:
            pp_table: Table name (e.g. _PpTable.bus)
            pp_idx: PandaPower component identifier
            name: optional name for the index

        Returns:
            the generated IDs
        """
        key = (pp_table, name)
        if key in self.idx_lookup:
            raise KeyError(f"Indexes for '{key}' already exist!")
        n_objects = len(pp_idx)
        pgm_idx = np.arange(self.next_idx, self.next_idx + n_objects).astype(np.int32)
        self.idx[key] = pd.Series(pgm_idx, index=pp_idx)
        self.idx_lookup[key] = pd.Series(pp_idx, index=pgm_idx)
        self.next_idx += n_objects
        return pgm_idx

    def _get_pgm_ids(
        self,
        pp_table: str,
        pp_idx: pd.Series | np.ndarray | None = None,
        name: str | None = None,
    ) -> pd.Series:
        """
        Get numerical power-grid-model IDs for a PandaPower component

        Args:
            pp_table: Table name (e.g. _PpTable.bus)
            pp_idx: PandaPower component identifier
            name: optional name for the index

        Returns:
            the power-grid-model IDs if they were previously generated
        """
        key = (pp_table, name)
        if key not in self.idx:
            raise KeyError(f"No indexes have been created for '{pp_table}' (name={name})!")
        if pp_idx is None:
            return self.idx[key]
        return self.idx[key][pp_idx]

    def _get_pp_ids(
        self,
        pp_table: str,
        pgm_idx: pd.Series | None = None,
        name: str | None = None,
    ) -> pd.Series:
        """
        Get numerical PandaPower IDs for a PandaPower component

        Args:
            pp_table: Table name (e.g. _PpTable.bus)
            pgm_idx: power-grid-model component identifier
            name: optional name for the index

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
            the Network (e.g. _PpAttr.hv_bus, _PpAttr.i0_percent)

        Returns:
            the "tap size" of Transformers
        """
        tap_side_hv = np.array(pp_trafo[_PpAttr.tap_side] == "hv")
        tap_side_lv = np.array(pp_trafo[_PpAttr.tap_side] == "lv")
        tap_step_multiplier = pp_trafo[_PpAttr.tap_step_percent] * (1e-2 * 1e3)

        tap_size = np.empty(shape=len(pp_trafo), dtype=np.float64)
        tap_size[tap_side_hv] = tap_step_multiplier[tap_side_hv] * pp_trafo[_PpAttr.vn_hv_kv][tap_side_hv]
        tap_size[tap_side_lv] = tap_step_multiplier[tap_side_lv] * pp_trafo[_PpAttr.vn_lv_kv][tap_side_lv]

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
            the Network (e.g. _PpAttr.hv_bus, _PpAttr.i0_percent)

        Returns:
            the "tap size" of Three Winding Transformers
        """
        tap_side_hv = np.array(pp_3wtrafo[_PpAttr.tap_side] == "hv")
        tap_side_mv = np.array(pp_3wtrafo[_PpAttr.tap_side] == "mv")
        tap_side_lv = np.array(pp_3wtrafo[_PpAttr.tap_side] == "lv")

        tap_step_multiplier = pp_3wtrafo[_PpAttr.tap_step_percent] * (1e-2 * 1e3)

        tap_size = np.empty(shape=len(pp_3wtrafo), dtype=np.float64)
        tap_size[tap_side_hv] = tap_step_multiplier[tap_side_hv] * pp_3wtrafo[_PpAttr.vn_hv_kv][tap_side_hv]
        tap_size[tap_side_mv] = tap_step_multiplier[tap_side_mv] * pp_3wtrafo[_PpAttr.vn_mv_kv][tap_side_mv]
        tap_size[tap_side_lv] = tap_step_multiplier[tap_side_lv] * pp_3wtrafo[_PpAttr.vn_lv_kv][tap_side_lv]

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
        switch_states = (
            component[["index", bus]]
            .merge(
                switches,
                how="left",
                left_on=["index", bus],
                right_on=[_PpAttr.element, _PpAttr.bus],
            )
            .set_index(component.index)[_PpAttr.closed]
        )

        # no need to fill na because bool(NaN) == True
        if Version(version("pandas")) >= Version("3.0.0"):
            return pd.Series(switch_states.astype(bool))
        return pd.Series(switch_states.astype(bool, copy=False))

    def get_switch_states(self, pp_table: str) -> pd.DataFrame:
        """
        Return switch states of either Lines or Transformers

        Args:
            pp_table: Table name (e.g. _PpTable.bus)

        Returns:
            the switch states of either Lines or Transformers
        """
        if pp_table == _PpTable.line:
            element_type = "l"
            bus1 = _PpAttr.from_bus
            bus2 = _PpAttr.to_bus
        elif pp_table == _PpTable.trafo:
            element_type = "t"
            bus1 = _PpAttr.hv_bus
            bus2 = _PpAttr.lv_bus
        else:
            raise KeyError(f"Can't get switch states for {pp_table}")

        component = self.pp_input_data[pp_table].copy()
        component["index"] = component.index

        # Select the appropriate switches and columns
        pp_switches = self.pp_input_data[_PpTable.switch]
        pp_switches = pp_switches[pp_switches[_PpAttr.et] == element_type]
        pp_switches = pp_switches[[_PpAttr.element, _PpAttr.bus, _PpAttr.closed]]

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
        bus1 = _PpAttr.hv_bus
        bus2 = _PpAttr.mv_bus
        bus3 = _PpAttr.lv_bus
        trafo3w["index"] = trafo3w.index

        # Select the appropriate switches and columns
        pp_switches = self.pp_input_data[_PpTable.switch]
        pp_switches = pp_switches[pp_switches[_PpAttr.et] == element_type]
        pp_switches = pp_switches[[_PpAttr.element, _PpAttr.bus, _PpAttr.closed]]

        # Join the switches with the three winding trafo three times, for the hv_bus, mv_bus and once for the lv_bus
        pp_1_switches = self.get_individual_switch_states(trafo3w[["index", bus1]], pp_switches, bus1)
        pp_2_switches = self.get_individual_switch_states(trafo3w[["index", bus2]], pp_switches, bus2)
        pp_3_switches = self.get_individual_switch_states(trafo3w[["index", bus3]], pp_switches, bus3)

        return pd.DataFrame(
            data={
                "side_1": pp_1_switches,
                "side_2": pp_2_switches,
                "side_3": pp_3_switches,
            },
            index=trafo3w.index,
        )

    def get_trafo_winding_types(self) -> pd.DataFrame:
        """
        This function extracts Transformers' "winding_type" attribute through "vector_group" attribut.

        Returns:
            the "from" and "to" winding types of a transformer
        """

        @lru_cache
        def vector_group_to_winding_types(vector_group: str) -> pd.Series:
            trafo_connection = parse_trafo_connection(vector_group)
            winding_from = get_winding(trafo_connection["winding_from"]).value
            winding_to = get_winding(trafo_connection["winding_to"]).value
            return pd.Series([winding_from, winding_to])

        trafo = self.pp_input_data[_PpTable.trafo]
        col_names = ["winding_from", "winding_to"]
        if _PpAttr.vector_group not in trafo:
            return pd.DataFrame(np.full(shape=(len(trafo), 2), fill_value=np.nan), columns=col_names)
        trafo = trafo[_PpAttr.vector_group].apply(vector_group_to_winding_types)
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
            trafo_connection = parse_trafo3_connection(vector_group)
            winding_1 = get_winding(trafo_connection["winding_1"]).value
            winding_2 = get_winding(trafo_connection["winding_2"]).value
            winding_3 = get_winding(trafo_connection["winding_3"]).value
            return pd.Series([winding_1, winding_2, winding_3])

        trafo3w = self.pp_input_data[_PpTable.trafo3w]
        col_names = ["winding_1", "winding_2", "winding_3"]
        if "vector_group" not in trafo3w:
            return pd.DataFrame(np.full(shape=(len(trafo3w), 3), fill_value=np.nan), columns=col_names)
        trafo3w = trafo3w["vector_group"].apply(vector_group_to_winding_types)
        trafo3w.columns = col_names
        return trafo3w

    def _get_pp_attr(
        self,
        table: str,
        attribute: str,
        expected_type: str | None = None,
        default: float | bool | str | None = None,
    ) -> np.ndarray:
        """
        Returns the selected PandaPower attribute from the selected PandaPower table.

        Args:
            table: Table name (e.g. _PpTable.bus)
            attribute: an attribute from the table (e.g "vn_kv")
            expected_type: optional expected type of the attribute
            default: optional default value for the attribute

        Returns:
            the selected PandaPower attribute from the selected PandaPower table
        """
        pp_component_data = self.pp_input_data[table]

        exp_dtype: str | type = "O"
        if expected_type is not None:
            exp_dtype = expected_type
        elif default is not None:
            exp_dtype = type(default)

        # If the attribute does not exist, return the default value
        # (assume that broadcasting is handled by the caller / numpy)
        if attribute not in pp_component_data:
            if default is None:
                raise KeyError(f"No '{attribute}' value for '{table}'.")
            return np.array([default], dtype=exp_dtype)

        attr_data = pp_component_data[attribute]

        # If any of the attribute values are missing, and a default is supplied, fill the nans with the default value
        nan_values = np.equal(attr_data, None) if attr_data.dtype is np.dtype("O") else np.isnan(attr_data)  # type: ignore

        if any(nan_values):
            attr_data = attr_data.fillna(value=default, inplace=False)

        return attr_data.to_numpy(dtype=exp_dtype, copy=True)

    def get_id(self, pp_table: str, pp_idx: int, name: str | None = None) -> int:
        """
        Get a numerical ID previously associated with the supplied table / index combination

        Args:
            pp_table: Table name (e.g. _PpTable.bus)
            pp_idx: PandaPower component identifier
            name: Optional component name (e.g. "internal_node")

        Returns:
            The associated id
        """
        return self.idx[(pp_table, name)][pp_idx]

    def lookup_id(self, pgm_id: int) -> dict[str, str | int]:
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
