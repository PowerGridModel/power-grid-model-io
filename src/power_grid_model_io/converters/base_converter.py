# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
import structlog


class BaseConverter:
    def __init__(self):
        self._log = structlog.getLogger(type(self).__name__)
