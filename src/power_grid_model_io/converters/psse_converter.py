# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
SIEMENS PSS®E RAWX Converter: Load data from JSON RAWX file and use a mapping file to convert the data to PGM
"""

import logging
from pathlib import Path

from power_grid_model import DatasetType
from power_grid_model.data_types import Dataset, SingleDataset

from power_grid_model_io.converters.base_converter import BaseConverter
from power_grid_model_io.data_stores.rawx_file_store import RawxFileStore
from power_grid_model_io.data_types import ExtraInfo, RawxData


class PsseConverter(BaseConverter[RawxData]):
    """SIEMENS PSS®E RAWX Converter: Load data from JSON RAWX file and use a mapping file to convert the data to PGM"""

    __slots__ = ("pgm_input_data", )

    def __init__(
            self,
            source_file: Path | str | None = None,
            log_level: int = logging.INFO,
        ):
            """
            Prepare some member variables and optionally load a mapping file

            Args:
                mapping_file: A yaml file containing the mapping.
            """

            source = (
                RawxFileStore(file_path=Path(source_file))
                if source_file
                else None
            )
            super().__init__(source=source, log_level=log_level)
            self.pgm_input_data: SingleDataset = {}

    def _parse_data(
        self,
        data: RawxData,
        data_type: DatasetType,
        _: ExtraInfo | None = None
    ) -> Dataset:
        """

        Args:
            data (RawxData): _description_
            data_type (DatasetType, optional): _description_. Defaults to DatasetType.input.
            extra_info (ExtraInfo | None, optional): _description_. Defaults to None.

        Returns:
            Dataset: _description_
        """
        self.pgm_input_data = {}
        self.pgm_input_data["data"] = data

        # Convert
        if data_type == DatasetType.input:
            # self._create_input_data()
            pass
        else:
            raise ValueError(f"Data type: '{data_type}' is not implemented")

        return self.pgm_input_data

    def _serialize_data(self, _: Dataset, __: ExtraInfo | None):
          raise NotImplementedError("Serialize method not implemented for PsseConverter")
