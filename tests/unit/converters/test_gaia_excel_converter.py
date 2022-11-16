# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
from unittest.mock import patch

import pandas as pd
import pytest

from power_grid_model_io.converters.gaia_excel_converter import GaiaExcelConverter


@pytest.fixture
def converter():
    converter = GaiaExcelConverter()
    return converter


def test_initialization():
    with pytest.raises(FileNotFoundError, match="No Gaia Excel mapping available for language 'abcde'"):
        GaiaExcelConverter(language="abcde")

    with patch(
        "power_grid_model_io.converters.tabular_converter.TabularConverter.set_mapping_file"
    ) as mock_set_mapping_file:
        GaiaExcelConverter()
        mock_set_mapping_file.assert_called_once()

    with patch("power_grid_model_io.converters.gaia_excel_converter.GaiaExcelFileStore") as MockFileStore:
        GaiaExcelConverter()
        MockFileStore.assert_not_called()

        GaiaExcelConverter(source_file="source_file")
        MockFileStore.assert_called_once()
