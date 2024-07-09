# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
from pathlib import Path
from unittest.mock import patch

import pytest

from power_grid_model_io.converters.vision_excel_converter import DEFAULT_MAPPING_FILE, VisionExcelConverter
from power_grid_model_io.utils.excel_ambiguity_checker import ExcelAmbiguityChecker


@pytest.fixture
def converter() -> VisionExcelConverter:
    # Arrange
    converter = VisionExcelConverter()
    converter._get_id("Nodes", {"Number": 1}, None)  # node: 0
    converter._get_id("Cables", {"Number": 1}, None)  # branch: 1
    converter._get_id("Links", {"Number": 1}, None)  # branch: 2
    converter._get_id("Reactance coils", {"Number": 1}, None)  # branch: 3
    converter._get_id("Special transformers", {"Number": 1}, None)  # branch: 4
    converter._get_id("Transformer loads", {"Node.Number": 1, "Subnumber": 2}, "transformer")  # virtual: 5
    converter._get_id("Transformer loads", {"Node.Number": 1, "Subnumber": 2}, "internal_node")  # virtual:  6
    converter._get_id("Transformer loads", {"Node.Number": 1, "Subnumber": 2}, "load")  # virtual:  7
    converter._get_id("Transformer loads", {"Node.Number": 1, "Subnumber": 2}, "generation")  # virtual:  8
    converter._get_id("Transformer loads", {"Node.Number": 1, "Subnumber": 2}, "pv_generation")  # virtual:  9
    converter._get_id("Sources", {"Node.Number": 1, "Subnumber": 2}, None)  # appliance: 10
    converter._get_id("Synchronous generators", {"Node.Number": 1, "Subnumber": 2}, None)  # appliance: 11
    converter._get_id("Wind turbines", {"Node.Number": 1, "Subnumber": 2}, None)  # appliance: 12
    converter._get_id("Loads", {"Node.Number": 1, "Subnumber": 2}, None)  # appliance: 13
    converter._get_id("Zigzag transformers", {"Node.Number": 1, "Subnumber": 2}, None)  # appliance: 14
    converter._get_id("Pvs", {"Node.Number": 1, "Subnumber": 2}, None)  # appliance: 15
    return converter


@pytest.mark.parametrize("language", ["en", "nl"])
def test_mapping_files_exist(language: str):
    vf = Path(str(DEFAULT_MAPPING_FILE).format(language=language))
    assert vf.exists()


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


def test_id_lookup_exceptions():
    # Arrange
    converter = VisionExcelConverter()
    converter._id_reference = None

    # Act / Assert
    with pytest.raises(ValueError, match=r"Missing .* for VisionExcelConverter\.get_node_id\(\)"):
        converter.get_node_id(number=0)

    with pytest.raises(ValueError, match=r"Missing .* VisionExcelConverter\.get_branch_id\(\)"):
        converter.get_branch_id(table="", number=0)

    with pytest.raises(ValueError, match=r"Missing .* VisionExcelConverter\.get_appliance_id\(\)"):
        converter.get_appliance_id(table="", node_number=0, sub_number=0)

    with pytest.raises(ValueError, match=r"Missing .* VisionExcelConverter\.get_virtual_id\(\)"):
        converter.get_virtual_id(table="", obj_name="", node_number=0, sub_number=0)


def test_get_node_id(converter: VisionExcelConverter):
    # Act / Assert
    assert converter.get_node_id(number=1) == 0

    with pytest.raises(KeyError):
        converter.get_node_id(number=2)


def test_get_branch_id(converter: VisionExcelConverter):
    # Act / Assert
    assert converter.get_branch_id(table="Cables", number=1) == 1
    assert converter.get_branch_id(table="Links", number=1) == 2
    assert converter.get_branch_id(table="Reactance coils", number=1) == 3
    assert converter.get_branch_id(table="Special transformers", number=1) == 4

    with pytest.raises(KeyError):
        converter.get_branch_id(table="Dummy", number=1)

    with pytest.raises(KeyError):
        converter.get_branch_id(table="Cables", number=2)


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

    with pytest.raises(KeyError):
        converter.get_virtual_id(table="Dummy", obj_name="transformer", node_number=1, sub_number=2)

    with pytest.raises(KeyError):
        converter.get_virtual_id(table="Transformer loads", obj_name="Dummy", node_number=1, sub_number=2)

    with pytest.raises(KeyError):
        converter.get_virtual_id(table="Transformer loads", obj_name="transformer", node_number=2, sub_number=2)

    with pytest.raises(KeyError):
        converter.get_virtual_id(table="Transformer loads", obj_name="transformer", node_number=1, sub_number=3)


def test_get_appliance_id(converter: VisionExcelConverter):
    # Act / Assert
    assert converter.get_appliance_id(table="Sources", node_number=1, sub_number=2) == 10
    assert converter.get_appliance_id(table="Synchronous generators", node_number=1, sub_number=2) == 11
    assert converter.get_appliance_id(table="Wind turbines", node_number=1, sub_number=2) == 12
    assert converter.get_appliance_id(table="Loads", node_number=1, sub_number=2) == 13
    assert converter.get_appliance_id(table="Zigzag transformers", node_number=1, sub_number=2) == 14
    assert converter.get_appliance_id(table="Pvs", node_number=1, sub_number=2) == 15

    with pytest.raises(KeyError):
        converter.get_appliance_id(table="Dummy", node_number=1, sub_number=2)

    with pytest.raises(KeyError):
        converter.get_appliance_id(table="Sources", node_number=2, sub_number=2)

    with pytest.raises(KeyError):
        converter.get_appliance_id(table="Sources", node_number=1, sub_number=3)


def test_ambiguity_in_vision_excel():
    ambiguious_test_file = Path(__file__).parents[2] / "data" / "vision" / "excel_ambiguity_check_data.xlsx"
    excel_file_checker = ExcelAmbiguityChecker(file_path=ambiguious_test_file.as_posix())
    res, _ = excel_file_checker.check_ambiguity()
    assert res == True
