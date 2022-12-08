# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import pytest

from power_grid_model_io.data_stores.base_data_store import BaseDataStore


def test_abstract_methods():
    with pytest.raises(TypeError, match=r"with abstract methods load, save"):
        BaseDataStore()
