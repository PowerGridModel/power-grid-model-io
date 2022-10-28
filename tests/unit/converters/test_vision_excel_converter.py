# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
import pytest
from unittest.mock import patch

from power_grid_model_io.converters.vision_excel_converter import VisionExcelConverter


MAPPING_FILE = Path(__file__).parent / "test_data/mapping.yaml"


@pytest.fixture
def converter():
    converter = VisionExcelConverter()
    return converter


def test_initialization():
    with pytest.raises(FileNotFoundError, match="No Vision Excel mapping available for language 'abcde'"):
        VisionExcelConverter(language="abcde")

    with patch("power_grid_model_io.converters.tabular_converter.TabularConverter.set_mapping_file") as mock_set_mapping_file:
        VisionExcelConverter()
        mock_set_mapping_file.assert_called_once()

    with patch("power_grid_model_io.data_stores.vision_excel_file_store.VisionExcelFileStore") as MockFileStore:
        VisionExcelConverter()
        MockFileStore.assert_not_called()

        # TODO: fix lines below
        # VisionExcelConverter(source_file="source_file")
        # MockFileStore.assert_called_once()


def test_converter__id_lookup():
    pass
