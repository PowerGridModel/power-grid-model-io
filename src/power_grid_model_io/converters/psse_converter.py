# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
SIEMENS PSS®E RAWX Converter: Load data from JSON RAWX file and convert the data to PGM
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
from power_grid_model import AttributeType, ComponentType, DatasetType, initialize_array
from power_grid_model.data_types import Dataset, SingleDataset

from power_grid_model_io._enum import PsseTable as _PsseTable
from power_grid_model_io.converters.base_converter import BaseConverter
from power_grid_model_io.data_stores.rawx_file_store import RawxFileStore
from power_grid_model_io.data_types import ExtraInfo, RawxData


class PsseConverter(BaseConverter[RawxData]):
    """
    SIEMENS PSS®E RAWX Converter: Load data from JSON RAWX file and convert the data to PGM
    """

    __slots__ = ("pgm_input_data", "psse_input_data")

    def __init__(
        self,
        source_file: Path | str | None = None,
        log_level: int = logging.INFO,
    ):
        """
        Prepare some member variables

        Args:
            source_file: A rawx file containing the psse data.
        """

        source = (
            RawxFileStore(file_path=Path(source_file))
            if source_file
            else None
        )
        super().__init__(source=source, log_level=log_level)
        self.pgm_input_data: SingleDataset = {}
        self.psse_input_data: RawxData = RawxData()
        self.idx: dict[tuple[str, str | None], pd.Series] = {}
        self.idx_lookup: dict[tuple[str, str | None], pd.Series] = {}
        self.next_idx = 0

    def _parse_data(
        self,
        data: RawxData,
        data_type: DatasetType,
        _: ExtraInfo | None = None
    ) -> Dataset:
        """
        Setup for conversion from PSS/E to power-grid-model.

        Args:
            data: RawxData, PSS/E data in the format RawxData.
            data_type: power-grid-model data type, i.e. DatasetType.input.

        Returns:
            Converted power-grid-model data
        """
        self.pgm_input_data = {}

        self.psse_input_data = data

        # Convert
        if data_type == DatasetType.input:
            self._create_input_data()
        else:
            raise ValueError(f"Data type: '{data_type}' is not implemented")

        return self.pgm_input_data

    def _serialize_data(self, _: Dataset, __: ExtraInfo | None):
          raise NotImplementedError("Serialize method not implemented for PsseConverter")

    def _create_input_data(self):
        """
        Performs the conversion from PSS/E to power-grid-model by calling individual conversion functions
        """
        self._create_pgm_input_nodes()
        self._create_pgm_input_lines()
        # self._create_pgm_input_sources()
        # self._create_pgm_input_sym_loads()
        # self._create_pgm_input_shunts()
        # self._create_pgm_input_transformers()
        # self._create_pgm_input_sgens()
        # self._create_pgm_input_gens()
        # self._create_pgm_input_three_winding_transformers()
        # self._create_pgm_input_links()
        # self._create_pgm_input_asym_loads()
        # self._create_pgm_input_asym_gens()
        # self._create_pgm_input_wards()
        # self._create_pgm_input_motors()
        # self._create_pgm_input_storages()
        # self._create_pgm_input_impedances()
        # self._create_pgm_input_xwards()
        # self._create_pgm_input_dclines()

    def _create_pgm_input_nodes(self):
        """
        This function converts a Bus Dataframe of PSS/E data to a power-grid-model Node input array.
        The converted array is added to pgm_input_data slot.
        """
        busses = self.psse_input_data[_PsseTable.bus]

        if busses.empty:
            return

        if ComponentType.node in self.pgm_input_data:
            raise ValueError("Node component already exists in pgm_input_data")

        pgm_nodes = initialize_array(
            data_type=DatasetType.input, component_type=ComponentType.node, shape=len(busses)
        )
        pgm_nodes[AttributeType.id] = self._generate_ids(_PsseTable.bus, busses.ibus)
        pgm_nodes[AttributeType.u_rated] = self.psse_input_data.get_column(
             table_name=_PsseTable.bus,
             column_name="baskv"
        ).to_numpy() * 1e3

        self.pgm_input_data[ComponentType.node] = pgm_nodes

    def _create_pgm_input_lines(self):
        """
        This function converts a acline table of PSSE to a power-grid-model Line input array.
        """
        psse_lines = self._input_data[_PsseTable.acline]

        if psse_lines.empty:
            return

        if ComponentType.line in self.pgm_input_data:
            raise ValueError("Line component already exists in pgm_input_data")

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

    def _generate_ids(self, psse_table: _PsseTable, psse_idx: pd.Index, name: str | None = None) -> np.ndarray:
        """
        Generate numerical power-grid-model IDs for a psse component

        Args:
            psse_table: Table name (e.g. _PsseTable.bus)
            psse_idx: Psse component identifier
            name: optional name for the index

        Returns:
            the generated IDs
        """
        key = (psse_table, name)
        if key in self.idx_lookup:
            raise KeyError(f"Indexes for '{key}' already exist!")
        n_objects = len(psse_idx)
        pgm_idx = np.arange(self.next_idx, self.next_idx + n_objects).astype(np.int32)
        self.idx[key] = pd.Series(pgm_idx, index=psse_idx)
        self.idx_lookup[key] = pd.Series(psse_idx, index=pgm_idx)
        self.next_idx += n_objects
        return pgm_idx

    def _get_pgm_ids(
        self,
        psse_table: _PsseTable,
        psse_idx: pd.Series | np.ndarray | None = None,
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
        key = (psse_table, name)
        if key not in self.idx:
            raise KeyError(f"No indexes have been created for '{psse_table}' (name={name})!")
        if psse_idx is None:
            return self.idx[key]
        return self.idx[key][psse_idx]
