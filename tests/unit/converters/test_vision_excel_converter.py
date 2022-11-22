# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
from unittest.mock import patch

import pytest

from power_grid_model_io.converters.vision_excel_converter import VisionExcelConverter


@pytest.fixture
def converter() -> VisionExcelConverter:
    # Arrange
    converter = VisionExcelConverter()
    converter.get_id("Nodes", {"Number": 1})  # node: 0
    converter.get_id("Cables", {"Number": 1})  # branch: 1
    converter.get_id("Links", {"Number": 1})  # branch: 2
    converter.get_id("Reactance coils", {"Number": 1})  # branch: 3
    converter.get_id("Special transformers", {"Number": 1})  # branch: 4
    converter.get_id(["Transformer loads", "transformer"], {"Node.Number": 1, "Subnumber": 2})  # virtual: 5
    converter.get_id(["Transformer loads", "internal_node"], {"Node.Number": 1, "Subnumber": 2})  # virtual:  6
    converter.get_id(["Transformer loads", "load"], {"Node.Number": 1, "Subnumber": 2})  # virtual:  7
    converter.get_id(["Transformer loads", "generation"], {"Node.Number": 1, "Subnumber": 2})  # virtual:  8
    converter.get_id(["Transformer loads", "pv_generation"], {"Node.Number": 1, "Subnumber": 2})  # virtual:  9
    converter.get_id("Sources", {"Node.Number": 1, "Subnumber": 2})  # appliance: 10
    converter.get_id("Synchronous generators", {"Node.Number": 1, "Subnumber": 2})  # appliance: 11
    converter.get_id("Wind turbines", {"Node.Number": 1, "Subnumber": 2})  # appliance: 12
    converter.get_id("Loads", {"Node.Number": 1, "Subnumber": 2})  # appliance: 13
    converter.get_id("Zigzag transformers", {"Node.Number": 1, "Subnumber": 2})  # appliance: 14
    converter.get_id("Pvs", {"Node.Number": 1, "Subnumber": 2})  # appliance: 15
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


def test_get_node_id(converter: VisionExcelConverter):
    # Act / Assert
    assert converter.get_node_id(number=1) == 0


def test_get_branch_id(converter: VisionExcelConverter):
    # Act / Assert
    assert converter.get_branch_id(table="Cables", number=1) == 1
    assert converter.get_branch_id(table="Links", number=1) == 2
    assert converter.get_branch_id(table="Reactance coils", number=1) == 3
    assert converter.get_branch_id(table="Special transformers", number=1) == 4


def test_get_virtual_id(converter: VisionExcelConverter):
    # Act / Assert
    assert converter.get_virtual_id(table="Transformer loads", obj_name="transformer", node_number=1, sub_number=2) == 5
    assert (
        converter.get_virtual_id(table="Transformer loads", obj_name="internal_node", node_number=1, sub_number=2) == 6
    )
    assert converter.get_virtual_id(table="Transformer loads", obj_name="load", node_number=1, sub_number=2) == 7
    assert converter.get_virtual_id(table="Transformer loads", obj_name="generation", node_number=1, sub_number=2) == 8
    assert (
        converter.get_virtual_id(table="Transformer loads", obj_name="pv_generation", node_number=1, sub_number=2) == 9
    )


def test_get_appliance_id(converter: VisionExcelConverter):
    # Act / Assert
    assert converter.get_appliance_id(table="Sources", node_number=1, sub_number=2) == 10
    assert converter.get_appliance_id(table="Synchronous generators", node_number=1, sub_number=2) == 11
    assert converter.get_appliance_id(table="Wind turbines", node_number=1, sub_number=2) == 12
    assert converter.get_appliance_id(table="Loads", node_number=1, sub_number=2) == 13
    assert converter.get_appliance_id(table="Zigzag transformers", node_number=1, sub_number=2) == 14
    assert converter.get_appliance_id(table="Pvs", node_number=1, sub_number=2) == 15
