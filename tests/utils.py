# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import sys
from copy import copy, deepcopy
from typing import Any, Dict, List, Optional, Tuple, Union
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from pandas.core.generic import NDFrame


def dict_in_dict(needle: Dict[str, Any], data: Dict[str, Any]) -> bool:
    return all(item in data.items() for item in needle.items())


def assert_struct_array_equal(actual: np.ndarray, expected: np.ndarray):
    """
    Compare two structured numpy arrays by converting them to pandas DataFrames first
    """
    pd.testing.assert_frame_equal(pd.DataFrame(actual), pd.DataFrame(expected))


def assert_log_exists(
    capture: List[Dict[str, Any]], log_level: Optional[str] = None, event: Optional[str] = None, **kwargs
):
    if log_level is not None:
        kwargs["log_level"] = log_level
    if event is not None:
        kwargs["event"] = event
    if not any(dict_in_dict(kwargs, log_line) for log_line in capture):
        capture = deepcopy(capture)
        print(
            "Logs:\n"
            + "\n".join(f"{i}: [{log.pop('log_level')}] {log.pop('event')} {log}" for i, log in enumerate(capture)),
            file=sys.stderr,
        )
        raise KeyError(f"Log {kwargs} does not exist")


def assert_log_match(capture: Dict[str, Any], level: Optional[str] = None, event: Optional[str] = None, **kwargs):
    if level is not None:
        kwargs["log_level"] = level
    if event is not None:
        kwargs["event"] = event
    if not dict_in_dict(kwargs, capture):
        capture = deepcopy(capture)
        print(f"Log:\n[{capture.pop('log_level')}] {capture.pop('event')} {capture}", file=sys.stderr)
        raise KeyError(f"Expected log {kwargs} does not match actual log {capture}")


def idx_to_str(idx: Union[str, int, slice, Tuple[Union[int, slice], ...]]) -> str:
    if isinstance(idx, tuple):
        return ", ".join(idx_to_str(i) for i in idx)
    if isinstance(idx, slice):
        start = idx.start or ""
        stop = idx.stop or ""
        step = idx.step or ""
        if idx.step is None:
            return f"{start}:{stop}"
        return f"{start}:{stop}:{step}"
    return repr(idx)


class MockFn:
    __slots__ = ["fn", "args", "kwargs", "postfix"]

    def __init__(self, fn: str, *args, **kwargs):
        self.fn = fn
        self.args = args
        self.kwargs = kwargs
        self.postfix = ""

    def is_operator(self) -> bool:
        return self.fn in {"+", "&", "/", "*"}

    def __copy__(self):
        mock_fn = MockFn(self.fn, *self.args, **self.kwargs)
        mock_fn.postfix = self.postfix
        return mock_fn

    def __add__(self, other):
        return MockFn("+", self, other)

    def __and__(self, other):
        return MockFn("&", self, other)

    def __truediv__(self, other):
        return MockFn("/", self, other)

    def __mul__(self, other):
        return MockFn("*", self, other)

    def __eq__(self, other):
        if not isinstance(other, MockFn):
            return False

        def eq(left, right) -> bool:
            if isinstance(left, pd.DataFrame):
                if left.columns != right.columns:
                    return False
            if isinstance(left, pd.Series):
                if left.name != right.name:
                    return False
            if isinstance(left, NDFrame):
                return (left == right).all()
            return left == right

        return (
            eq(self.fn, other.fn)
            and all(eq(a, b) for a, b in zip(self.args, other.args))
            and self.kwargs.keys() == other.kwargs.keys()
            and all(eq(self.kwargs[k], other.kwargs[k]) for k in self.kwargs)
            and self.postfix == other.postfix
        )

    def __repr__(self):
        if self.is_operator():
            return f"{self.args[0]} {self.fn} {self.args[1]}"
        args = [repr(arg) for arg in self.args] + [f"{key}={repr(val)}" for key, val in self.kwargs.items()]
        return f"{self.fn}({', '.join(args)}){self.postfix}"

    def __getattr__(self, item):
        mock_fn = copy(self)
        mock_fn.postfix += f".{item}"
        return mock_fn

    def __getitem__(self, item):
        mock_fn = copy(self)
        mock_fn.postfix += f"[{idx_to_str(item)}]"
        return mock_fn


class MockVal(MockFn):
    def __init__(self, value: Any):
        super().__init__(fn=value)

    def __copy__(self):
        mock_val = MockVal(value=self.fn)
        mock_val.postfix = self.postfix
        return mock_val

    def __repr__(self):
        return repr(self.fn)


class MockDf:
    __slots__ = ["empty", "index", "shape"]

    def __init__(self, shape: Union[int, Tuple[int, ...]]):
        self.shape = shape
        self.empty = shape == 0 or (isinstance(shape, tuple) and (len(shape) == 0 or shape[0] == 0))
        self.index = MagicMock()

    def __len__(self):
        if isinstance(self.shape, int):
            return self.shape
        if isinstance(self.shape, tuple) and len(self.shape) > 0:
            return self.shape[0]
        return 0

    def __getitem__(self, item: str):
        return MockVal(pd.Series(name=item, dtype=np.float64))
