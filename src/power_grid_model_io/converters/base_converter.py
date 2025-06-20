# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
Abstract converter class
"""

import logging
from abc import ABC, abstractmethod
from typing import Generic, Optional, Tuple, TypeVar

import structlog
from power_grid_model import DatasetType
from power_grid_model.data_types import Dataset, SingleDataset

from power_grid_model_io.data_stores.base_data_store import BaseDataStore
from power_grid_model_io.data_types import ExtraInfo
from power_grid_model_io.utils.auto_id import AutoID

T = TypeVar("T")


class BaseConverter(Generic[T], ABC):
    """Abstract converter class"""

    def __init__(
        self,
        source: Optional[BaseDataStore[T]] = None,
        destination: Optional[BaseDataStore[T]] = None,
        log_level: int = logging.INFO,
    ):
        """
        Initialize a logger
        """
        self._logger = logging.getLogger(f"{__name__}_{id(self)}")
        self._logger.setLevel(log_level)
        self._log = structlog.wrap_logger(self._logger, wrapper_class=structlog.make_filtering_bound_logger(log_level))
        self._source = source
        self._destination = destination
        self._auto_id = AutoID()

    def load_input_data(
        self, data: Optional[T] = None, make_extra_info: bool = True
    ) -> Tuple[SingleDataset, ExtraInfo]:
        """Load input data and extra info

        Note: You shouldn't have to overwrite this method. Check _parse_data() instead.

        Args:
          data: Optionally supply data in source format. If no data is supplied, it is loaded from self._source
          make_extra_info: For efficiency reasons, one can disable the creation of extra_info.
        Returns:

        """

        data = self._load_data(data)
        extra_info: ExtraInfo = {}
        parsed_data = self._parse_data(
            data=data, data_type=DatasetType.input, extra_info=extra_info if make_extra_info else None
        )
        if isinstance(parsed_data, list):
            raise TypeError("Input data can not be batch data")
        return parsed_data, extra_info

    def load_update_data(self, data: Optional[T] = None) -> Dataset:
        """Load update data

        Note: You shouldn't have to overwrite this method. Check _parse_data() instead.

        Args:
          data: Optional[T]:  (Default value = None)

        Returns:

        """
        data = self._load_data(data)
        return self._parse_data(data=data, data_type=DatasetType.update, extra_info=None)

    def load_sym_output_data(self, data: Optional[T] = None) -> Dataset:
        """Load symmetric output data

        Note: You shouldn't have to overwrite this method. Check _parse_data() instead.

        Args:
          data: Optional[T]:  (Default value = None)

        Returns:

        """
        data = self._load_data(data)
        return self._parse_data(data=data, data_type=DatasetType.sym_output, extra_info=None)

    def load_asym_output_data(self, data: Optional[T] = None) -> Dataset:
        """Load asymmetric output data

        Note: You shouldn't have to overwrite this method. Check _parse_data() instead.

        Args:
          data: Optional[T]:  (Default value = None)

        Returns:

        """
        data = self._load_data(data)
        return self._parse_data(data=data, data_type=DatasetType.asym_output, extra_info=None)

    def load_sc_output_data(self, data: Optional[T] = None) -> Dataset:
        """Load sc output data

        Note: You shouldn't have to overwrite this method. Check _parse_data() instead.

        Args:
          data: Optional[T]:  (Default value = None)

        Returns:

        """
        data = self._load_data(data)
        return self._parse_data(data=data, data_type=DatasetType.sc_output, extra_info=None)

    def convert(self, data: Dataset, extra_info: Optional[ExtraInfo] = None) -> T:
        """Convert input/update/(a)sym_output data and optionally extra info.

        Note: You shouldn't have to overwrite this method. Check _serialize_data() instead.

        Args:
          data: Dataset:
          extra_info: Optional[ExtraInfo]:  (Default value = None)

        Returns:

        """
        return self._serialize_data(data=data, extra_info=extra_info)

    def save(
        self,
        data: Dataset,
        extra_info: Optional[ExtraInfo] = None,
        destination: Optional[BaseDataStore[T]] = None,
    ) -> None:
        """Save input/update/(a)sym_output data and optionally extra info.

        Note: You shouldn't have to overwrite this method. Check _serialize_data() instead.

        Args:
          data: Dataset:
          extra_info: Optional[ExtraInfo]:  (Default value = None)
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

    def set_log_level(self, log_level: int) -> None:
        """
        Set the log level

        Args:
          log_level: int:
        """
        self._logger.setLevel(log_level)
        self._log = structlog.wrap_logger(self._logger, wrapper_class=structlog.make_filtering_bound_logger(log_level))

    def get_log_level(self) -> int:
        """
        Get the log level

        Returns:
          int:
        """
        return self._logger.getEffectiveLevel()

    def _load_data(self, data: Optional[T]) -> T:
        if data is not None:
            return data
        if self._source is not None:
            return self._source.load()
        raise ValueError("No data supplied!")

    @abstractmethod  # pragma: nocover
    def _parse_data(self, data: T, data_type: DatasetType, extra_info: Optional[ExtraInfo]) -> Dataset:
        pass

    @abstractmethod  # pragma: nocover
    def _serialize_data(self, data: Dataset, extra_info: Optional[ExtraInfo]) -> T:
        pass
