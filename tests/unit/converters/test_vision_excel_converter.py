import pytest

from power_grid_model_io.converters.vision_excel_converter import VisionExcelConverter

@pytest.fixture
def converter():
    converter = VisionExcelConverter()
    return converter


def test_converter__id_lookup():
    pass
