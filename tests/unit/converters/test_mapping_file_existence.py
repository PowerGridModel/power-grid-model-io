# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0


from pathlib import Path

from power_grid_model_io.converters.gaia_excel_converter import DEFAULT_MAPPING_FILE as gaia_mapping
from power_grid_model_io.converters.vision_excel_converter import DEFAULT_MAPPING_FILE as vision_mapping


def test_mapping_files_exist():
    gf = Path(str(gaia_mapping).format(language="en"))
    vf = Path(str(vision_mapping).format(language="en"))
    assert gf.exists()
    assert vf.exists()
