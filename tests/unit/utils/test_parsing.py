# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from power_grid_model_io.utils.parsing import parse_node_ref, parse_trafo3_connection, parse_trafo_connection


def test_trafo_connection__pos():
    assert parse_trafo_connection("Dyn") == {"winding_from": "D", "winding_to": "yn", "clock_number": None}
    assert parse_trafo_connection("Yyn") == {"winding_from": "Y", "winding_to": "yn", "clock_number": None}
    assert parse_trafo_connection("Yzn") == {"winding_from": "Y", "winding_to": "zn", "clock_number": None}
    assert parse_trafo_connection("YNy") == {"winding_from": "YN", "winding_to": "y", "clock_number": None}
    assert parse_trafo_connection("Dy5") == {"winding_from": "D", "winding_to": "y", "clock_number": "5"}
    assert parse_trafo_connection("Dy11") == {"winding_from": "D", "winding_to": "y", "clock_number": "11"}


def test_trafo_connection__neg():
    assert not parse_trafo_connection("Xyn")
    assert not parse_trafo_connection("yyn")
    assert not parse_trafo_connection("YZN")
    assert not parse_trafo_connection("YNx")
    assert not parse_trafo_connection("Dy13")
    assert not parse_trafo_connection("Dy-1")


def test_trafo3_connection__pos():
    assert parse_trafo3_connection("Dyn1yn1") == {
        "winding_1": "D",
        "winding_2": "yn",
        "clock_12": "1",
        "winding_3": "yn",
        "clock_13": "1",
    }
    assert parse_trafo3_connection("Yyn2d4") == {
        "winding_1": "Y",
        "winding_2": "yn",
        "clock_12": "2",
        "winding_3": "d",
        "clock_13": "4",
    }
    assert parse_trafo3_connection("Yzn4y0") == {
        "winding_1": "Y",
        "winding_2": "zn",
        "clock_12": "4",
        "winding_3": "y",
        "clock_13": "0",
    }
    assert parse_trafo3_connection("YNd1z0") == {
        "winding_1": "YN",
        "winding_2": "d",
        "clock_12": "1",
        "winding_3": "z",
        "clock_13": "0",
    }
    assert parse_trafo3_connection("Dy1y5") == {
        "winding_1": "D",
        "winding_2": "y",
        "clock_12": "1",
        "winding_3": "y",
        "clock_13": "5",
    }
    assert parse_trafo3_connection("Dy5d11") == {
        "winding_1": "D",
        "winding_2": "y",
        "clock_12": "5",
        "winding_3": "d",
        "clock_13": "11",
    }


def test_trafo3_connection__neg():
    assert not parse_trafo3_connection("Xynd")
    assert not parse_trafo3_connection("ydyn")
    assert not parse_trafo3_connection("DYZN")
    assert not parse_trafo3_connection("YNxd")
    assert not parse_trafo3_connection("Dyd13")
    assert not parse_trafo3_connection("DyD10")
    assert not parse_trafo3_connection("Dynd-1")
    assert not parse_trafo3_connection("Dyn+5d-1")
    assert not parse_trafo3_connection("Dy1d13")
    assert not parse_trafo3_connection("Da1d10")
    assert not parse_trafo3_connection("Xy1d10")


def test_node_ref__pos():
    assert parse_node_ref("node")
    assert parse_node_ref("from_node")
    assert parse_node_ref("to_node")
    assert parse_node_ref("node_1")
    assert parse_node_ref("node_2")
    assert parse_node_ref("node_3")


def test_node_ref__neg():
    assert not parse_node_ref("from_bla")
    assert not parse_node_ref("a_node_node_b")
    assert not parse_node_ref("nodes")
    assert not parse_node_ref("anode")
    assert not parse_node_ref("immunodeficient")
