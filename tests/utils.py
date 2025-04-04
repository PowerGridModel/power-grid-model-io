# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import sys
from copy import copy, deepcopy
from itertools import chain
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Tuple, Union
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
from pandas.core.generic import NDFrame


def contains(needle: Mapping[str, Any], data: Mapping[str, Any]) -> bool:
    return all(item in data.items() for item in needle.items())


def assert_struct_array_equal(actual: np.ndarray, expected: Union[np.ndarray, pd.DataFrame, List[Dict[str, Any]]]):
    """
    Compare two structured numpy arrays by converting them to pandas DataFrames first
    """
    pd.testing.assert_frame_equal(pd.DataFrame(actual), pd.DataFrame(expected), check_dtype=False)


def assert_log_exists(
    capture: List[MutableMapping[str, Any]], log_level: Optional[str] = None, event: Optional[str] = None, **kwargs
):
    if log_level is not None:
        kwargs["log_level"] = log_level
    if event is not None:
        kwargs["event"] = event
    if not any(contains(kwargs, log_line) for log_line in capture):
        capture = deepcopy(capture)
        print(
            "Logs:\n"
            + "\n".join(f"{i}: [{log.pop('log_level')}] {log.pop('event')} {log}" for i, log in enumerate(capture)),
            file=sys.stderr,
        )
        raise KeyError(f"Log {kwargs} does not exist")


def assert_log_match(
    capture: MutableMapping[str, Any], level: Optional[str] = None, event: Optional[str] = None, **kwargs
):
    if level is not None:
        kwargs["log_level"] = level
    if event is not None:
        kwargs["event"] = event
    if not contains(kwargs, capture):
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

    __array_struct__ = np.array([]).__array_struct__
    __array_prepare__ = np.array([]).__array_prepare__

    def __init__(self, fn: str, *args, **kwargs):
        self.fn = fn
        self.args = list(args)
        self.kwargs = kwargs
        self.postfix = ""

    @staticmethod
    def _is_operator(obj):
        return isinstance(obj, MockFn) and obj.is_operator()

    @staticmethod
    def _apply_operator(fn: str, left: Any, right: Any):
        if MockFn._is_operator(left) and left.fn == fn:
            obj = copy(left)
        else:
            obj = MockFn(fn, left)
        if MockFn._is_operator(right):
            if (
                obj.fn == "+"
                and right.fn == "+"
                or obj.fn == "-"
                and right.fn == "+"
                or obj.fn == "*"
                and right.fn == "*"
                or obj.fn == "/"
                and right.fn == "*"
                or obj.fn == "&"
                and right.fn == "&"
            ):
                obj.args += right.args
                return obj
        obj.args += [right]
        return obj

    def is_operator(self) -> bool:
        return (
            isinstance(self.fn, str) and self.fn in {"+", "-", "*", "/", "&", "%"} and not (self.kwargs or self.postfix)
        )

    def is_commutative(self) -> bool:
        return isinstance(self.fn, str) and self.fn in {"+", "*", "&", "|"} and not (self.kwargs or self.postfix)

    def __copy__(self):
        mock_fn = MockFn(self.fn, *self.args, **self.kwargs)
        mock_fn.postfix = self.postfix
        return mock_fn

    def __neg__(self):
        return MockFn("-", self)

    def __add__(self, other):
        return MockFn._apply_operator("+", self, other)

    def __radd__(self, other):
        return MockFn._apply_operator("+", other, self)

    def __and__(self, other):
        return MockFn._apply_operator("&", self, other)

    def __or__(self, other):
        return MockFn._apply_operator("|", self, other)

    def __truediv__(self, other):
        return MockFn._apply_operator("/", self, other)

    def __rtruediv__(self, other):
        return MockFn._apply_operator("/", other, self)

    def __sub__(self, other):
        return MockFn._apply_operator("-", self, other)

    def __rsub__(self, other):
        return MockFn._apply_operator("-", other, self)

    def __mul__(self, other):
        return MockFn._apply_operator("*", self, other)

    def __rmul__(self, other):
        return MockFn._apply_operator("*", other, self)

    def __mod__(self, other):
        return MockFn._apply_operator("%", self, other)

    def __round__(self):
        return MockFn("round", self)

    def __eq__(self, other):
        if not isinstance(other, MockFn):
            return False

        def isnan(x: Any):
            if isinstance(x, np.ndarray) and x.size == 0:
                return False
            try:
                return np.isnan(x)
            except TypeError:
                return False

        def eq(left, right) -> bool:
            if type(left) != type(right):
                return False
            if isinstance(left, pd.DataFrame) and left.columns != right.columns:
                return False
            if isinstance(left, pd.Series) and left.name != right.name:
                return False
            if isinstance(left, NDFrame):
                return (left == right).all()
            if isinstance(right, NDFrame):
                return False

            if isinstance(left, np.ndarray) and left.size == 0:
                return isinstance(right, np.ndarray) and right.size == 0

            if left == right:
                return True

            left_nans = isnan(left)
            right_nans = isnan(right)

            if np.any(left_nans != right_nans):
                return False

            infinite = np.logical_and(left_nans, right_nans)
            if isinstance(left, np.ndarray):
                finite = np.logical_not(infinite)
                return bool(np.all(np.equal(left[finite], right[finite])))

            if isinstance(infinite, np.ndarray) and np.any(infinite):
                return True

            return infinite

        if not eq(self.fn, other.fn):
            return False

        if self.is_commutative():
            return set(self.args) == set(other.args)

        return (
            all(eq(a, b) for a, b in zip(self.args, other.args))
            and self.kwargs.keys() == other.kwargs.keys()
            and all(eq(self.kwargs[k], other.kwargs[k]) for k in self.kwargs)
            and self.postfix == other.postfix
        )

    def __hash__(self):
        return hash((self.fn, tuple(self.args), tuple(self.kwargs.items()), self.postfix))

    def __repr__(self):
        if self.is_operator():
            return "(" + f" {self.fn} ".join(repr(arg) for arg in self.args) + ")"
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

    def __call__(self, *args, **kwargs):
        mock_fn = copy(self)
        arguments = ", ".join(chain(map(repr, args), map(lambda k, v: f"{k}={repr(v)}", kwargs.items())))
        mock_fn.postfix += f"({arguments})"
        return mock_fn

    def __setitem__(self, item, val):
        mock_fn = copy(self)
        mock_fn.postfix += f"[{idx_to_str(item)}]={val}"


class MockVal(MockFn):
    def __init__(self, value: Any):
        super().__init__(fn=value)

    def __copy__(self):
        mock_val = MockVal(value=self.fn)
        mock_val.postfix = self.postfix
        return mock_val

    def __repr__(self):
        return str(self.fn)


class MockDf:
    __slots__ = ["columns", "empty", "index", "shape"]

    def __init__(self, shape: Union[int, Tuple[int, ...]]):
        self.shape = shape
        self.empty = shape == 0 or (isinstance(shape, tuple) and (len(shape) == 0 or shape[0] == 0))
        self.index = MagicMock(name="DataFrame.index")
        self.columns = MagicMock(name="DataFrame.columns")

    def __len__(self):
        if isinstance(self.shape, int):
            return self.shape
        if isinstance(self.shape, tuple) and len(self.shape) > 0:
            return self.shape[0]
        return 0

    def __getitem__(self, item: str):
        return MockVal(pd.Series(name=item, dtype=np.float64))


class MockExcelFile:
    def __init__(self, data: Dict[str, pd.DataFrame]):
        self.data = data

    @property
    def sheet_names(self) -> List[str]:
        return list(self.data.keys())

    def parse(self, sheet_name: str, **_kwargs) -> pd.DataFrame:
        return self.data[sheet_name]


class MockTqdm:
    """To use: for x in tqdm(iterable)"""

    def __init__(self, iterable=None, **kwargs):
        self.iterable = iterable

    def __iter__(self):
        for item in self.iterable:
            yield item
