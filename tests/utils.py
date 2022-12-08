# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
import sys
from typing import Any, Dict, List, Optional


def _array_in_array(needle: Dict[str, Any], data: Dict[str, Any]) -> bool:
    return all(item in data.items() for item in needle.items())


def assert_log_exists(
    capture: List[Dict[str, Any]], log_level: Optional[str] = None, event: Optional[str] = None, **kwargs
):
    if log_level is not None:
        kwargs["log_level"] = log_level
    if event is not None:
        kwargs["event"] = event
    if not any(_array_in_array(kwargs, log_line) for log_line in capture):
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
    if not _array_in_array(kwargs, capture):
        print(f"Log:\n[{capture.pop('log_level')}] {capture.pop('event')} {capture}", file=sys.stderr)
        raise KeyError(f"Expected log {kwargs} does not match actual log {capture}")
