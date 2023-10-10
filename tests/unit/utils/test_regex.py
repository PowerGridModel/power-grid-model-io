# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from power_grid_model_io.utils.regex import TRAFO3_CONNECTION_RE, get_trafo_connection, is_node_ref


def test_trafo_connection__pos():
    assert get_trafo_connection("Dyn") == {"winding_from": "D", "winding_to": "yn", "clock_number": None}
    assert get_trafo_connection("Yyn") == {"winding_from": "Y", "winding_to": "yn", "clock_number": None}
    assert get_trafo_connection("Yzn") == {"winding_from": "Y", "winding_to": "zn", "clock_number": None}
    assert get_trafo_connection("YNy") == {"winding_from": "YN", "winding_to": "y", "clock_number": None}
    assert get_trafo_connection("Dy5") == {"winding_from": "D", "winding_to": "y", "clock_number": "5"}
    assert get_trafo_connection("Dy11") == {"winding_from": "D", "winding_to": "y", "clock_number": "11"}


def test_trafo_connection__neg():
    assert not get_trafo_connection("Xyn")
    assert not get_trafo_connection("yyn")
    assert not get_trafo_connection("YZN")
    assert not get_trafo_connection("YNx")
    assert not get_trafo_connection("Dy13")
    assert not get_trafo_connection("Dy-1")


def test_trafo3_connection__pos():
    assert TRAFO3_CONNECTION_RE.fullmatch("Dyn1yn1").groups() == ("D", "yn", "1", "yn", "1")
    assert TRAFO3_CONNECTION_RE.fullmatch("Yyn2d4").groups() == ("Y", "yn", "2", "d", "4")
    assert TRAFO3_CONNECTION_RE.fullmatch("Yzn4y0").groups() == ("Y", "zn", "4", "y", "0")
    assert TRAFO3_CONNECTION_RE.fullmatch("YNd1z0").groups() == ("YN", "d", "1", "z", "0")
    assert TRAFO3_CONNECTION_RE.fullmatch("Dy1y5").groups() == ("D", "y", "1", "y", "5")
    assert TRAFO3_CONNECTION_RE.fullmatch("Dy5d11").groups() == ("D", "y", "5", "d", "11")


def test_trafo3_connection__neg():
    assert not TRAFO3_CONNECTION_RE.fullmatch("Xynd")
    assert not TRAFO3_CONNECTION_RE.fullmatch("ydyn")
    assert not TRAFO3_CONNECTION_RE.fullmatch("DYZN")
    assert not TRAFO3_CONNECTION_RE.fullmatch("YNxd")
    assert not TRAFO3_CONNECTION_RE.fullmatch("Dyd13")
    assert not TRAFO3_CONNECTION_RE.fullmatch("DyD10")
    assert not TRAFO3_CONNECTION_RE.fullmatch("Dynd-1")
    assert not TRAFO3_CONNECTION_RE.fullmatch("Dyn+5d-1")
    assert not TRAFO3_CONNECTION_RE.fullmatch("Dy1d13")
    assert not TRAFO3_CONNECTION_RE.fullmatch("Da1d10")
    assert not TRAFO3_CONNECTION_RE.fullmatch("Xy1d10")


def test_node_ref__pos():
    assert is_node_ref("node")
    assert is_node_ref("from_node")
    assert is_node_ref("to_node")
    assert is_node_ref("node_1")
    assert is_node_ref("node_2")
    assert is_node_ref("node_3")


def test_node_ref__neg():
    assert not is_node_ref("from_bla")
    assert not is_node_ref("a_node_node_b")
    assert not is_node_ref("nodes")
    assert not is_node_ref("anode")
    assert not is_node_ref("immunodeficient")
