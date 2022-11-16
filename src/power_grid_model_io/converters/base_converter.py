# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Abstract converter class
"""
from abc import ABC, abstractmethod
from typing import Dict, Generic, List, Mapping, Optional, Tuple, TypeVar, Union

import structlog
from power_grid_model.data_types import Dataset, SingleDataset

from power_grid_model_io.data_stores.base_data_store import BaseDataStore
from power_grid_model_io.data_types import ExtraInfoLookup
from power_grid_model_io.utils.auto_id import AutoID

T = TypeVar("T")


class BaseConverter(Generic[T], ABC):
    """Abstract converter class"""

    def __init__(self, source: Optional[BaseDataStore[T]] = None, destination: Optional[BaseDataStore[T]] = None):
        """
        Initialize a logger
        """
        self._log = structlog.get_logger(type(self).__name__)
        self._source = source
        self._destination = destination
        self._auto_id = AutoID()

    def load_input_data(self, data: Optional[T] = None) -> Tuple[SingleDataset, ExtraInfoLookup]:
        """Load input data and extra info

        Note: You shouldn't have to overwrite this method. Check _parse_data() instead.

        Args:
          data: Optional[T]:  (Default value = None)

        Returns:

        """
        data = self._load_data(data)
        extra_info: ExtraInfoLookup = {}
        data = self._parse_data(data=data, data_type="input", extra_info=extra_info)
        if isinstance(data, list):
            raise TypeError("Input data can not be batch data")
        return data, extra_info

    def load_update_data(self, data: Optional[T] = None) -> Dataset:
        """Load update data

        Note: You shouldn't have to overwrite this method. Check _parse_data() instead.

        Args:
          data: Optional[T]:  (Default value = None)

        Returns:

        """
        data = self._load_data(data)
        return self._parse_data(data=data, data_type="update", extra_info=None)

    def load_sym_output_data(self, data: Optional[T] = None) -> Dataset:
        """Load symmetric output data

        Note: You shouldn't have to overwrite this method. Check _parse_data() instead.

        Args:
          data: Optional[T]:  (Default value = None)

        Returns:

        """
        data = self._load_data(data)
        return self._parse_data(data=data, data_type="sym_output", extra_info=None)

    def load_asym_output_data(self, data: Optional[T] = None) -> Dataset:
        """Load asymmetric output data

        Note: You shouldn't have to overwrite this method. Check _parse_data() instead.

        Args:
          data: Optional[T]:  (Default value = None)

        Returns:

        """
        data = self._load_data(data)
        return self._parse_data(data=data, data_type="asym_output", extra_info=None)

    def convert(self, data: Dataset, extra_info: Optional[ExtraInfoLookup] = None) -> T:
        """Convert input/update/(a)sym_output data and optionally extra info.

        Note: You shouldn't have to overwrite this method. Check _serialize_data() instead.

        Args:
          data: Dataset:
          extra_info: Optional[ExtraInfoLookup]:  (Default value = None)

        Returns:

        """
        return self._serialize_data(data=data, extra_info=extra_info)

    def save(
        self,
        data: Dataset,
        extra_info: Optional[ExtraInfoLookup] = None,
        destination: Optional[BaseDataStore[T]] = None,
    ) -> None:
        """Save input/update/(a)sym_output data and optionally extra info.

        Note: You shouldn't have to overwrite this method. Check _serialize_data() instead.

        Args:
          data: Dataset:
          extra_info: Optional[ExtraInfoLookup]:  (Default value = None)
          destination: Optional[BaseDataStore[T]]:  (Default value = None)

        Returns:

        """
        data_converted = self.convert(data=data, extra_info=extra_info)
        if destination is not None:
            destination.save(data=data_converted)
        elif self._destination is not None:
            self._destination.save(data=data_converted)
        else:
            raise ValueError("No destination supplied!")

    def _load_data(self, data: Optional[T]) -> T:
        if data is not None:
            return data
        if self._source is not None:
            return self._source.load()
        raise ValueError("No data supplied!")

    def get_id(self, name: Union[str, List[str], Tuple[str, ...]], key: Mapping[str, int]) -> int:
        """
        Get a unique numerical ID for the name / key combination

        Args:
            name: Component name (e.g. "Node" or ["Transformer", "Internal node"])
            key: Component identifier (e.g. {"name": "node1"} or {"number": 1, "sub_number": 2})

        Returns: A unique id
        """
        if isinstance(name, list):
            name = tuple(name)
        return self._auto_id(item=(name, tuple(key.items())))

    def lookup_id(self, pgm_id: int) -> Tuple[Union[str, List[str]], Dict[str, int]]:
        """
        Retrieve the original name / key combination of a pgm object

        Args:
            pgm_id: a unique numerical ID

        Returns: The original name / key combination
        """
        name, key = self._auto_id[pgm_id]
        if isinstance(name, tuple):
            name = list(name)
        return name, dict(key)

    @abstractmethod  # pragma: nocover
    def _parse_data(self, data: T, data_type: str, extra_info: Optional[ExtraInfoLookup]) -> Dataset:
        pass

    @abstractmethod  # pragma: nocover
    def _serialize_data(self, data: Dataset, extra_info: Optional[ExtraInfoLookup]) -> T:
        pass
