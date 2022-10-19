# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from power_grid_model_io.converters.vision_excel_converter import VisionExcelConverter


@pytest.fixture
def converter():
    converter = VisionExcelConverter()
    return converter


def test_converter__id_lookup():
    pass
