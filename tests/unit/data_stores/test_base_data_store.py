# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from power_grid_model_io.data_stores.base_data_store import BaseDataStore


def test_abstract_methods():
    with pytest.raises(TypeError, match=r"abstract methods .*load.* .*save.*"):
        BaseDataStore()
