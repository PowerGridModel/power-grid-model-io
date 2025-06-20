# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
Tabular data mapping helper class
"""

from typing import Dict, Generator, List, Tuple

import structlog

AttributeValue = int | float | str | Dict | List
InstanceAttributes = Dict[str, AttributeValue]
Components = Dict[str, InstanceAttributes | List[InstanceAttributes]]
Tables = Dict[str, Components]


class TabularMapping:
    """
    Tabular data mapping helper class
    """

    def __init__(self, mapping: Tables, logger=None):
        if logger is None:
            self._log = structlog.get_logger(f"{__name__}_{id(self)}")
        else:
            self._log = logger
        self._log.debug("Set tabular mapping", n_mappings=len(mapping))
        self._mapping: Tables = mapping

    def tables(self) -> Generator[str, None, None]:
        """
        Return the names of the tables (as a generator)

        Yields:
            table_name
        """
        return (key for key in self._mapping.keys())

    def instances(self, table: str) -> Generator[Tuple[str, InstanceAttributes], None, None]:
        """
        Return instance definitions (as a generator)

        Yields:
            component_name, instance_attribute_mapping
        """
        table_mapping = self._mapping.get(table, {})
        if not isinstance(table_mapping, dict):
            raise TypeError(
                f"Invalid table mapping for {table}; expected a dictionary got {type(table_mapping).__name__}"
            )
        for component, instances in table_mapping.items():
            if not isinstance(instances, list):
                yield component, instances
            else:
                for instance in instances:
                    yield component, instance
