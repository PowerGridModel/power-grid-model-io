# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
from typing import Dict, Generator, List, Tuple, Union

import structlog

AttributeValue = Union[int, float, str, Dict, List]
InstanceAttributes = Dict[str, AttributeValue]
Components = Dict[str, Union[InstanceAttributes, List[InstanceAttributes]]]
Tables = Dict[str, Components]


class TabularMapping:
    def __init__(self, mapping: Tables):
        self._log = structlog.get_logger(type(self).__name__)
        self._log.debug(f"Set tabular mapping", n_mappings=len(mapping))
        self._mapping: Tables = mapping

    def tables(self) -> Generator[str, None, None]:
        return (key for key in self._mapping.keys())

    def instances(self, table: str) -> Generator[Tuple[str, InstanceAttributes], None, None]:
        for component, instances in self._mapping.get(table, {}).items():
            if not isinstance(instances, list):
                yield component, instances
            else:
                for instance in instances:
                    yield component, instance
