# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Abstract converter class
"""
from abc import ABC, abstractmethod
from typing import Generic, Optional, Tuple, TypeVar

import structlog
from power_grid_model.data_types import Dataset, SingleDataset

from power_grid_model_io.data_stores.base_data_store import BaseDataStore
from power_grid_model_io.data_types import ExtraInfoLookup

T = TypeVar("T")


class BaseConverter(Generic[T], ABC):
    """
    Abstract converter class
    """

    def __init__(self):
        """
        Initialize a logger
        """
        self._log = structlog.get_logger(type(self).__name__)

    def load_input_data(self, data: T) -> Tuple[SingleDataset, ExtraInfoLookup]:
        """
        Load input data and extra info

        Note: You shouldn't have to overwrite this method. Check _parse_data() instead.
        """
        extra_info: ExtraInfoLookup = {}
        data = self._parse_data(data=data, data_type="input", extra_info=extra_info)
        if isinstance(data, list):
            raise TypeError("Input data can not be batch data")
        return data, extra_info

    def load_update_data(self, data: T) -> Dataset:
        """
        Load update data

        Note: You shouldn't have to overwrite this method. Check _parse_data() instead.
        """
        return self._parse_data(data=data, data_type="update", extra_info=None)

    def load_sym_output_data(self, data: T) -> Dataset:
        """
        Load symmetric output data

        Note: You shouldn't have to overwrite this method. Check _parse_data() instead.
        """
        return self._parse_data(data=data, data_type="sym_output", extra_info=None)

    def load_asym_output_data(self, data: T) -> Dataset:
        """
        Load asymmetric output data

        Note: You shouldn't have to overwrite this method. Check _parse_data() instead.
        """
        return self._parse_data(data=data, data_type="asym_output", extra_info=None)

    def convert(self, data: Dataset, extra_info: Optional[ExtraInfoLookup] = None) -> T:
        """
        Convert input/update/(a)sym_output data and optionally extra info.

        Note: You shouldn't have to overwrite this method. Check _serialize_data() instead.
        """
        return self._serialize_data(data=data, extra_info=extra_info)

    @abstractmethod  # pragma: nocover
    def _parse_data(self, data: T, data_type: str, extra_info: Optional[ExtraInfoLookup] = None) -> Dataset:
        pass

    @abstractmethod  # pragma: nocover
    def _serialize_data(self, data: Dataset, extra_info: Optional[ExtraInfoLookup] = None) -> T:
        pass
