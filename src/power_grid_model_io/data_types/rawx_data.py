# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
The RawX class is a wrapper around Dict[str, pd.DataFrame | np.ndarray],
which supports unit conversions and value substitutions
"""

import numpy as np
import pandas as pd

from power_grid_model_io.data_types import BaseTabularData, LazyDataFrame


class RawxData(BaseTabularData):
    """
    The RawX class is a wrapper around Dict[str, pd.DataFrame | np.ndarray],
    which supports unit conversions and value substitutions
    """

    def __init__(
        self,
        logger=None,
        **tables: pd.DataFrame | np.ndarray | LazyDataFrame,
    ):
        """_summary_

        Args:
            logger (optional): A structlog logger to use for logging. If None, a default logger will be created.
            **tables: A collection of pandas DataFrames and/or numpy structured arrays
        """
        super().__init__(logger, **tables)
