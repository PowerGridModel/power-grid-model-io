# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Abstract converter class
"""
from abc import ABC, abstractmethod
from typing import Dict, Generic, Mapping, Optional, Tuple, TypeVar, Union

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

    def _get_id(self, table: str, key: Mapping[str, int], name: Optional[str]) -> int:
        """
        Get a unique numerical ID for the supplied name / key combination

        Args:
            table: Table name (e.g. "Nodes")
            key: Component identifier (e.g. {"name": "node1"} or {"number": 1, "sub_number": 2})
            name: Optional component name (e.g. "internal_node")

        Returns: A unique id
        """
        auto_id_key = (table, tuple(sorted(key.items())), name)
        return self._auto_id(item=(table, key, name), key=auto_id_key)

    def get_id(self, table: str, key: Mapping[str, int], name: Optional[str] = None) -> int:
        """
        Get a the numerical ID previously associated with the supplied name / key combination

        Args:
            table: Table name (e.g. "Nodes")
            key: Component identifier (e.g. {"name": "node1"} or {"number": 1, "sub_number": 2})
            name: Optional component name (e.g. "internal_node")

        Returns: The associated id
        """
        auto_id_key = (table, tuple(sorted(key.items())), name)
        if auto_id_key not in self._auto_id:
            raise KeyError((table, key, name))
        return self._auto_id(item=(table, key, name), key=auto_id_key)

    def lookup_id(self, pgm_id: int) -> Dict[str, Union[str, Dict[str, int]]]:
        """
        Retrieve the original name / key combination of a pgm object

        Args:
            pgm_id: a unique numerical ID

        Returns: The original name / key combination
        """
        table, key, name = self._auto_id[pgm_id]
        reference = {"table": table}
        if name is not None:
            reference["name"] = name
        reference["key"] = key
        return reference

    @abstractmethod  # pragma: nocover
    def _parse_data(self, data: T, data_type: str, extra_info: Optional[ExtraInfoLookup]) -> Dataset:
        pass

    @abstractmethod  # pragma: nocover
    def _serialize_data(self, data: Dataset, extra_info: Optional[ExtraInfoLookup]) -> T:
        pass
