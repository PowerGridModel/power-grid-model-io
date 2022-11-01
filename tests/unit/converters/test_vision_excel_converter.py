# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from power_grid_model_io.converters.vision_excel_converter import VisionExcelConverter

MAPPING_FILE = Path(__file__).parent / "test_data/mapping.yaml"


@pytest.fixture
def converter():
    converter = VisionExcelConverter()
    return converter


def test_initialization():
    with pytest.raises(FileNotFoundError, match="No Vision Excel mapping available for language 'abcde'"):
        VisionExcelConverter(language="abcde")

    with patch(
        "power_grid_model_io.converters.tabular_converter.TabularConverter.set_mapping_file"
    ) as mock_set_mapping_file:
        VisionExcelConverter()
        mock_set_mapping_file.assert_called_once()

    with patch("power_grid_model_io.converters.vision_excel_converter.VisionExcelFileStore") as MockFileStore:
        VisionExcelConverter()
        MockFileStore.assert_not_called()

        VisionExcelConverter(source_file="source_file")
        MockFileStore.assert_called_once()


def test_converter__id_lookup(converter: VisionExcelConverter):
    row_1 = pd.Series([4.0, 5.0, 6.0], index=["a.b", "c.d.e", "a.c"])
    row_2 = pd.Series([1.0, 5.0, 6.0], index=["a.b", "c.d.e", "a.c"])  # change in values
    row_3 = pd.Series([4.0, 5.0, 6.0], index=["z.b", "c.d.e", "a.c"])  # change in index

    assert converter._id_lookup(component="node", row=row_1) == 0
    assert converter._id_lookup(component="node", row=row_2) == 1
    assert converter._id_lookup(component="node", row=row_3) == 2
    assert converter._id_lookup(component="node", row=row_1) == 0

    assert converter._lookup._keys == {
        "node:b=4.0,c=6.0,e=5.0": 0,
        "node:b=1.0,c=6.0,e=5.0": 1,
        "node:c=6.0,e=5.0,b=4.0": 2,
    }
    assert converter._lookup._items[0] == {"component": "node", "row": {"b": 4.0, "c": 6.0, "e": 5.0}}
    assert converter._lookup._items[1] == {"component": "node", "row": {"b": 1.0, "c": 6.0, "e": 5.0}}
    assert converter._lookup._items[2] == {"component": "node", "row": {"c": 6.0, "e": 5.0, "b": 4.0}}
