# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
General regular expressions
"""

import re

TRAFO_CONNECTION_RE = re.compile(r"^(Y|YN|D|Z|ZN)(y|yn|d|z|zn)(\d|1[0-2])?$")
r"""
Regular expressions to the winding_from and winding_to codes and optionally the clock number:
^               Start of the string
(Y|YN|D|Z|ZN)   From winding type
(y|yn|d|z|zn)   To winding type
(\d|1[0-2])?    Optional clock number (0-12)
$               End of the string
"""

TRAFO3_CONNECTION_RE = re.compile(r"^(Y|YN|D|Z|ZN)(y|yn|d|z|zn)(\d|1[0-2])?(y|yn|d|z|zn)(\d|1[0-2])?$")
r"""
Regular expressions to the winding_1, winding_2 and winding_3 codes and optionally the clock number:
^               Start of the string
(Y|YN|D|Z|ZN)   First winding type
(y|yn|d|z|zn)   Second winding type
(\d|1[0-2])     Clock number (0-12)
(y|yn|d|z|zn)   Third winding type
(\d|1[0-2])     Clock number (0-12)
$               End of the string
"""

NODE_REF_RE = re.compile(r"^(.+_)?node(_.+)?$")
r"""
Regular expressions to match the word node with an optional prefix or suffix, e.g.:
    - node
    - from_node
    - node_1
^               Start of the string
(.+_)?          Optional prefix, ending with an underscore
node            The word 'node'
(_.+)?          Optional suffix, starting with in an underscore
$               End of the string
"""

PVS_EFFICIENCY_TYPE_RE = re.compile(r"[ ,..]1 pu: (95|97) %")
r"""
Regular expressions to match the efficiency type percentage at 1 pu, eg:
    - 0,1 pu: 93 %; 1 pu: 97 %
    - 0,1..1 pu: 95 %
1 pu            After 1 pu  '1 pu:'
(95|97)         95 or 97 % type
%               before  '%'
"""
