# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from enum import StrEnum


class _PandapowerTable(StrEnum):
    # Components
    bus = "bus"
    line = "line"
    ext_grid = "ext_grid"
    shunt = "shunt"
    sgen = "sgen"
    asymmetric_sgen = "asymmetric_sgen"
    load = "load"
    asymmetric_load = "asymmetric_load"
    trafo = "trafo"
    trafo3w = "trafo3w"
    switch = "switch"
    storage = "storage"
    impedance = "impedance"
    ward = "ward"
    xward = "xward"
    motor = "motor"
    dcline = "dcline"
    gen = "gen"
    # Result tables
    res_bus = "res_bus"
    res_line = "res_line"
    res_ext_grid = "res_ext_grid"
    res_shunt = "res_shunt"
    res_sgen = "res_sgen"
    res_asymmetric_sgen = "res_asymmetric_sgen"
    res_load = "res_load"
    res_asymmetric_load = "res_asymmetric_load"
    res_trafo = "res_trafo"
    res_trafo3w = "res_trafo3w"
    res_switch = "res_switch"
    res_storage = "res_storage"
    res_impedance = "res_impedance"
    res_ward = "res_ward"
    res_xward = "res_xward"
    res_motor = "res_motor"
    res_dcline = "res_dcline"
    res_gen = "res_gen"
