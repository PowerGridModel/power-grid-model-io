# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd


def assert_struct_array_equal(actual: np.ndarray, expected: np.ndarray):
    """
    Compare two structured numpy arrays by converting them to pandas DataFrames first
    """
    pd.testing.assert_frame_equal(pd.DataFrame(actual), pd.DataFrame(expected))
