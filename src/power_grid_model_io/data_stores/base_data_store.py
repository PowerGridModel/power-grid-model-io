# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import structlog

DataType = TypeVar("DataType")


class BaseDataStore(Generic[DataType], ABC):
    def __init__(self):
        self._log = structlog.get_logger(type(self).__name__)

    @abstractmethod  # pragma: no cover
    def load(self) -> DataType:
        pass

    @abstractmethod  # pragma: no cover
    def save(self, data: DataType) -> None:
        pass

    def _validate(self, data: DataType) -> None:

        # The data should be either a dictionary, or a (possibly empty) list of dictionaries
        if not isinstance(data, (dict, list)):
            raise TypeError(f"Invalid data type for {type(self).__name__}: {type(data).__name__}")

        if isinstance(data, list) and any(not isinstance(x, dict) for x in data):
            type_names = {type(x).__name__ for x in data}
            if len(type_names) == 1:
                type_str = type_names.pop()
            else:
                type_str = "Union[" + ", ".join(type_names) + "]"
            raise TypeError(f"Invalid data type for {type(self).__name__}: List[{type_str}]")
