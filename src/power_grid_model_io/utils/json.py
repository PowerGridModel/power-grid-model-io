# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Custom json encoding functionalities
"""

import json
from typing import IO, Any

import numpy as np


class JsonEncoder(json.JSONEncoder):
    """
    Custom JSON encoder for numpy types
    """

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


def compact_json_dump(data: Any, io_stream: IO[str], indent: int, max_level: int, level: int = 0):
    """Custom compact JSON writer that is intended to put data belonging to a single object on a single line.

    For example:
    {
        "node": [
            {"id": 0, "u_rated": 10500.0},
            {"id": 1, "u_rated": 10500.0},
        ],
        "line": [
            {"id": 2, "node_from": 0, "node_to": 1, ...}
        ]
    }

    The function is being called recursively, starting at level 0 and recursing until max_level is reached. It is
    basically a full json writer, but for efficiency reasons, on the last levels the native json.dump method is used.
    """

    # Let's define a 'tab' indent, depending on the level
    tab = " " * level * indent

    # If we are at the max_level, or the data simply doesn't contain any more levels, write the indent and serialize
    # the data on a single line.
    if level >= max_level or not isinstance(data, (list, dict)):
        io_stream.write(tab)
        json.dump(data, io_stream, indent=None, cls=JsonEncoder)
        return

    # We'll need the number of objects later on
    n_obj = len(data)

    # If the data is a list:
    # 1. start with an opening bracket
    # 2. dump each element in the list
    # 3. add a comma and a new line after each element, except for the last element, there we don't need a comma.
    # 4. finish with a closing bracket
    if isinstance(data, list):
        io_stream.write(tab + "[\n")
        for i, obj in enumerate(data, start=1):
            compact_json_dump(obj, io_stream, indent, max_level, level + 1)
            io_stream.write(",\n" if i < n_obj else "\n")
        io_stream.write(tab + "]")
        return

    # If the data is a dictionary:
    # 1. start with an opening curly bracket
    # 2. for each element: write it's key, plus a colon ':'
    # 3. if the next level would be the max_level, add a space and dump the element on a single,
    #    else add a new line before dumping the element recursively.
    # 4. add a comma and a new line after each element, except for the last element, there we don't need a comma.
    # 5. finish with a closing curly bracket
    io_stream.write(tab + "{\n")
    for i, (key, obj) in enumerate(data.items(), start=1):
        io_stream.write(tab + " " * indent + f'"{key}":')
        if level == max_level - 1 or not isinstance(obj, (list, dict)):
            io_stream.write(" ")
            json.dump(obj, io_stream, indent=None, cls=JsonEncoder)
        else:
            io_stream.write("\n")
            compact_json_dump(obj, io_stream, indent, max_level, level + 2)
        io_stream.write(",\n" if i < n_obj else "\n")
    io_stream.write(tab + "}\n")
