# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

from power_grid_model_io.converters import VisionExcelConverter
from power_grid_model_io.converters.vision_excel_converter import CONFIG_PATH
from power_grid_model_io.data_stores.base_data_store import DICT_KEY_NUMBER, LANGUAGE_EN, VISION_EXCEL_LAN_DICT
from power_grid_model_io.utils.uuid_excel_cvtr import convert_guid_vision_excel

terms_changed = {"Grounding1": "N1", "Grounding2": "N2", "Grounding3": "N3", "Load.Behaviour": "Behaviour"}

DATA_DIR = Path(__file__).parents[2] / "data" / "vision"
SOURCE_FILE = DATA_DIR / "vision_en_9_7.xlsx"
REFERENCE_FILE = DATA_DIR / "vision_en.xlsx"
MAPPING_FILE_EN_9_7 = CONFIG_PATH / "vision_en_9_7.yaml"


def test_convert_guid_vision_excel():
    new_file = convert_guid_vision_excel(
        SOURCE_FILE, number=VISION_EXCEL_LAN_DICT[LANGUAGE_EN][DICT_KEY_NUMBER], terms_changed=terms_changed
    )
    vision_cvtr_new = VisionExcelConverter(source_file=new_file, mapping_file=MAPPING_FILE_EN_9_7)
    vision_cvtr_ref = VisionExcelConverter(source_file=REFERENCE_FILE)

    data_new = vision_cvtr_new.load_input_data()
    data_ref = vision_cvtr_ref.load_input_data()

    assert len(data_new) == len(data_ref)
