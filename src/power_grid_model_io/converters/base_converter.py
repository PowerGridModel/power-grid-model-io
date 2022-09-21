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

    def __init__(self, source: Optional[BaseDataStore[T]] = None, destination: Optional[BaseDataStore[T]] = None):
        """
        Initialize a logger
        """
        self._log = structlog.get_logger(type(self).__name__)
        self._source = source
        self._destination = destination

    def load_input_data(self, data: Optional[T] = None) -> Tuple[SingleDataset, ExtraInfoLookup]:
        """
        Load input data and extra info

        Note: You shouldn't have to overwrite this method. Check _parse_data() instead.
        """
        data = self._load_data(data)
        extra_info: ExtraInfoLookup = {}
        data = self._parse_data(data=data, data_type="input", extra_info=extra_info)
        if isinstance(data, list):
            raise TypeError("Input data can not be batch data")
        return data, extra_info

    def load_update_data(self, data: Optional[T] = None) -> Dataset:
        """
        Load update data

        Note: You shouldn't have to overwrite this method. Check _parse_data() instead.
        """
        data = self._load_data(data)
        return self._parse_data(data=data, data_type="update", extra_info=None)

    def load_sym_output_data(self, data: Optional[T] = None) -> Dataset:
        """
        Load symmetric output data

        Note: You shouldn't have to overwrite this method. Check _parse_data() instead.
        """
        data = self._load_data(data)
        return self._parse_data(data=data, data_type="sym_output", extra_info=None)

    def load_asym_output_data(self, data: Optional[T] = None) -> Dataset:
        """
        Load asymmetric output data

        Note: You shouldn't have to overwrite this method. Check _parse_data() instead.
        """
        data = self._load_data(data)
        return self._parse_data(data=data, data_type="asym_output", extra_info=None)

    def convert(self, data: Dataset, extra_info: Optional[ExtraInfoLookup] = None) -> T:
        """
        Convert input/update/(a)sym_output data and optionally extra info.

        Note: You shouldn't have to overwrite this method. Check _serialize_data() instead.
        """
        return self._serialize_data(data=data, extra_info=extra_info)

    def save(
        self,
        data: Dataset,
        extra_info: Optional[ExtraInfoLookup] = None,
        destination: Optional[BaseDataStore[T]] = None,
    ) -> None:
        """
        Save input/update/(a)sym_output data and optionally extra info.

        Note: You shouldn't have to overwrite this method. Check _serialize_data() instead.
        """
        data = self.convert(data=data, extra_info=extra_info)
        if destination is not None:
            destination.save(data=data)
        elif self._destination is not None:
            self._destination.save(data=data)
        else:
            raise ValueError("No destination supplied!")

    def _load_data(self, data: Optional[T]) -> T:
        if data is not None:
            return data
        if self._source is not None:
            return self._source.load()
        raise ValueError("No data supplied!")

    @abstractmethod  # pragma: nocover
    def _parse_data(self, data: T, data_type: str, extra_info: Optional[ExtraInfoLookup] = None) -> Dataset:
        pass

    @abstractmethod  # pragma: nocover
    def _serialize_data(self, data: Dataset, extra_info: Optional[ExtraInfoLookup] = None) -> T:
        pass
