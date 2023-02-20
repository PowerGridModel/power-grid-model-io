# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from power_grid_model_io.utils.regex import NODE_REF_RE, TRAFO3_CONNECTION_RE, TRAFO_CONNECTION_RE


def test_trafo_connection__pos():
    assert TRAFO_CONNECTION_RE.fullmatch("Dyn").groups() == ("D", "yn", None)
    assert TRAFO_CONNECTION_RE.fullmatch("Yyn").groups() == ("Y", "yn", None)
    assert TRAFO_CONNECTION_RE.fullmatch("Yzn").groups() == ("Y", "zn", None)
    assert TRAFO_CONNECTION_RE.fullmatch("YNy").groups() == ("YN", "y", None)
    assert TRAFO_CONNECTION_RE.fullmatch("Dy5").groups() == ("D", "y", "5")
    assert TRAFO_CONNECTION_RE.fullmatch("Dy11").groups() == ("D", "y", "11")


def test_trafo_connection__neg():
    assert not TRAFO_CONNECTION_RE.fullmatch("Xyn")
    assert not TRAFO_CONNECTION_RE.fullmatch("yyn")
    assert not TRAFO_CONNECTION_RE.fullmatch("YZN")
    assert not TRAFO_CONNECTION_RE.fullmatch("YNx")
    assert not TRAFO_CONNECTION_RE.fullmatch("Dy13")
    assert not TRAFO_CONNECTION_RE.fullmatch("Dy-1")


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
    assert NODE_REF_RE.fullmatch("node")
    assert NODE_REF_RE.fullmatch("from_node")
    assert NODE_REF_RE.fullmatch("to_node")
    assert NODE_REF_RE.fullmatch("node_1")
    assert NODE_REF_RE.fullmatch("node_2")
    assert NODE_REF_RE.fullmatch("node_3")


def test_node_ref__neg():
    assert not NODE_REF_RE.fullmatch("nodes")
    assert not NODE_REF_RE.fullmatch("anode")
    assert not NODE_REF_RE.fullmatch("immunodeficient")
