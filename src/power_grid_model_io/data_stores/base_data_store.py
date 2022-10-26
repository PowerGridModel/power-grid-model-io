# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Abstract data store class
"""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import structlog

T = TypeVar("T")


class BaseDataStore(Generic[T], ABC):
    """
    Abstract data store class
    """

    def __init__(self):
        """
        Initialize a logger
        """
        self._log = structlog.get_logger(type(self).__name__)

    @abstractmethod  # pragma: no cover
    def load(self) -> T:
        """
        The method that loads the data from one or more sources and returns it in the specified format.
        Note that the load() method does not recieve a reference to the data source(s); i.e. the data souce(s)
        should be set in the constructor, or in a separate member method.
        """

    @abstractmethod  # pragma: no cover
    def save(self, data: T) -> None:
        """
        The method that saves the data to one or more destinations.
        Note that the save() method does not recieve a reference to the data destination(s); i.e. the data
        destination(s) should be set in the constructor, or in a separate member method.
        """
