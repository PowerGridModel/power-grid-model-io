# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
from unittest.mock import patch

import pandas as pd
import pytest

from power_grid_model_io.converters.vision_excel_converter import VisionExcelConverter


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

    assert converter._id_lookup(name="node", key=tuple(row_1)) == 0
    assert converter._id_lookup(name="node", key=tuple(row_2)) == 1
    assert converter._id_lookup(name="node", key=tuple(row_3)) == 0
    assert converter._id_lookup(name="node", key=tuple(row_1)) == 0

    assert converter._lookup[0] == ("node", (4.0, 5.0, 6.0))
    assert converter._lookup[1] == ("node", (1.0, 5.0, 6.0))
