# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import re

import pytest

from power_grid_model_io.utils.parsing import (
    is_node_ref,
    parse_node_ref,
    parse_trafo3_connection,
    parse_trafo_connection,
)


def test_parse_trafo_connection__pos():
    assert parse_trafo_connection("Dyn") == {"winding_from": "D", "winding_to": "yn", "clock": None}
    assert parse_trafo_connection("Yyn") == {"winding_from": "Y", "winding_to": "yn", "clock": None}
    assert parse_trafo_connection("Yzn") == {"winding_from": "Y", "winding_to": "zn", "clock": None}
    assert parse_trafo_connection("YNy") == {"winding_from": "YN", "winding_to": "y", "clock": None}
    assert parse_trafo_connection("Dy5") == {"winding_from": "D", "winding_to": "y", "clock": "5"}
    assert parse_trafo_connection("Dy11") == {"winding_from": "D", "winding_to": "y", "clock": "11"}


def test_parse_trafo_connection__neg():
    with pytest.raises(ValueError, match="Invalid transformer connection string: 'Xyn'"):
        parse_trafo_connection("Xyn")
    with pytest.raises(ValueError, match="Invalid transformer connection string: 'yyn'"):
        parse_trafo_connection("yyn")
    with pytest.raises(ValueError, match="Invalid transformer connection string: 'YZN'"):
        parse_trafo_connection("YZN")
    with pytest.raises(ValueError, match="Invalid transformer connection string: 'YNx'"):
        parse_trafo_connection("YNx")
    with pytest.raises(ValueError, match="Invalid transformer connection string: 'Dy13'"):
        parse_trafo_connection("Dy13")
    with pytest.raises(ValueError, match="Invalid transformer connection string: 'Dy-1'"):
        parse_trafo_connection("Dy-1")


def test_parse_trafo3_connection__pos():
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


def test_parse_trafo3_connection__neg():
    with pytest.raises(ValueError, match="Invalid three winding transformer connection string: 'Xynd'"):
        parse_trafo3_connection("Xynd")
    with pytest.raises(ValueError, match="Invalid three winding transformer connection string: 'ydyn'"):
        parse_trafo3_connection("ydyn")
    with pytest.raises(ValueError, match="Invalid three winding transformer connection string: 'DYZN'"):
        parse_trafo3_connection("DYZN")
    with pytest.raises(ValueError, match="Invalid three winding transformer connection string: 'YNxd'"):
        parse_trafo3_connection("YNxd")
    with pytest.raises(ValueError, match="Invalid three winding transformer connection string: 'Dyd13'"):
        parse_trafo3_connection("Dyd13")
    with pytest.raises(ValueError, match="Invalid three winding transformer connection string: 'DyD10'"):
        parse_trafo3_connection("DyD10")
    with pytest.raises(ValueError, match="Invalid three winding transformer connection string: 'Dynd-1'"):
        parse_trafo3_connection("Dynd-1")
    with pytest.raises(ValueError, match=re.escape("Invalid three winding transformer connection string: 'Dyn+5d-1'")):
        parse_trafo3_connection("Dyn+5d-1")
    with pytest.raises(ValueError, match="Invalid three winding transformer connection string: 'Dy1d13'"):
        parse_trafo3_connection("Dy1d13")
    with pytest.raises(ValueError, match="Invalid three winding transformer connection string: 'Da1d10'"):
        parse_trafo3_connection("Da1d10")
    with pytest.raises(ValueError, match="Invalid three winding transformer connection string: 'Xy1d10'"):
        parse_trafo3_connection("Xy1d10")


def test_parse_node_ref__pos():
    assert parse_node_ref("node")
    assert parse_node_ref("from_node")
    assert parse_node_ref("to_node")
    assert parse_node_ref("node_1")
    assert parse_node_ref("node_2")
    assert parse_node_ref("node_3")


def test_parse_node_ref__neg():
    with pytest.raises(ValueError, match="Invalid node reference string: 'from_bla'"):
        parse_node_ref("from_bla")
    with pytest.raises(ValueError, match="Invalid node reference string: 'a_node_node_b'"):
        parse_node_ref("a_node_node_b")
    with pytest.raises(ValueError, match="Invalid node reference string: 'nodes'"):
        parse_node_ref("nodes")
    with pytest.raises(ValueError, match="Invalid node reference string: 'anode'"):
        parse_node_ref("anode")
    with pytest.raises(ValueError, match="Invalid node reference string: 'immunodeficient'"):
        parse_node_ref("immunodeficient")


def test_is_node_ref__pos():
    assert is_node_ref("node")
    assert is_node_ref("from_node")
    assert is_node_ref("to_node")
    assert is_node_ref("node_1")
    assert is_node_ref("node_2")
    assert is_node_ref("node_3")


def test_is_node_ref__neg():
    assert not is_node_ref("from_bla")
    assert not is_node_ref("a_node_node_b")
    assert not is_node_ref("nodes")
    assert not is_node_ref("anode")
    assert not is_node_ref("immunodeficient")
