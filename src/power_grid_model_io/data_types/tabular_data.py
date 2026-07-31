# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
The TabularData class is a wrapper around Dict[str, pd.DataFrame | np.ndarray],
which supports unit conversions and value substitutions
"""

import contextlib
from typing import Protocol

import numpy as np
import pandas as pd

from power_grid_model_io.data_types import BaseTabularData, LazyDataFrame


class Closeable(Protocol):
    """Protocol for objects that expose a close() method."""

    def close(self) -> None:
        """Close the underlying resource."""


class TabularData(BaseTabularData):
    """
    The TabularData class is a wrapper around Dict[str, pd.DataFrame | np.ndarray],
    which supports unit conversions and value substitutions
    """

    def __init__(
        self,
        logger=None,
        open_readers=None,
        **tables: pd.DataFrame | np.ndarray | LazyDataFrame,
    ):
        """
        Tabular data can either be a collection of pandas DataFrames and/or numpy structured arrays.
        The key word arguments will define the keys of the data.

        tabular_data = TabularData(foo=foo_data)
        tabular_data["foo"] --> foo_data

        Args:
            logger (optional): A structlog logger to use for logging. If None, a default logger will be created.
            open_readers (optional): A list of readers to be closed when the TabularData instance is closed.
                These objects must implement a close() method.
            **tables: A collection of pandas DataFrames and/or numpy structured arrays
        """
        super().__init__(logger, **tables)
        self._open_readers: list[Closeable] = open_readers if open_readers is not None else []

    def __new__(cls, *args, **kwargs):  # noqa: ARG004
        instance = super().__new__(cls)
        instance._open_readers = None
        return instance

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        if self._open_readers:
            for reader in self._open_readers:
                with contextlib.suppress(Exception):
                    reader.close()
        self._open_readers = None
