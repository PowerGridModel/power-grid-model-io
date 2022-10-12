# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
These functions can be used in the mapping files to apply functions to pandapower data
"""


def positive_sequence_conductance(power: float, voltage: float) -> float:
    """
    Calculate positive sequence conductance as used in shunts
    """
    return power / (voltage * voltage)
