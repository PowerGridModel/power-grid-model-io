# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import structlog.testing

from power_grid_model_io.utils.uuid_excel_cvtr import convert_guid_vision_excel

terms_chaged = { "N1": "Grounding1",
                 "N2": "Grounding2",
                 "N3": "Grounding3",
                 }