# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
from typing import Dict, List, Union

import structlog

#           table     component attribute  value
Tables = Dict[str, Dict[str, Dict[str, Union[int, float, str, Dict, List]]]]


class TabularMapping:
    def __init__(self, mapping: Tables):
        self._log = structlog.get_logger(type(self).__name__)
        self._log.debug(f"Set tabular mapping", n_mappings=len(mapping))
        self._tables = mapping

    def get_mapping(self) -> Tables:
        return self._tables
