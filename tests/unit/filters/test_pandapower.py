# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from power_grid_model.enum import WindingType

from power_grid_model_io.filters.pandapower import (
    _split_string,
    _split_string_3wdg,
    get_3wdgtransformer_pk,
    get_3wdgtransformer_tap_size,
    get_3wdgtransformer_winding_1,
    get_3wdgtransformer_winding_2,
    get_3wdgtransformer_winding_3,
    get_transformer_clock,
    get_transformer_tap_size,
    get_transformer_winding_from,
    get_transformer_winding_to,
    positive_sequence_conductance,
)


def test_positive_sequence_conductance():
    assert positive_sequence_conductance(450.3, 10) == pytest.approx(4.503)
    assert positive_sequence_conductance(1000, 10) == pytest.approx(10)
    assert positive_sequence_conductance(20.2, 10) == pytest.approx(0.202)
    assert positive_sequence_conductance(225.3001, 15.01) == pytest.approx(1)
    with pytest.raises(TypeError):
        positive_sequence_conductance("20.2", "10")  # type: ignore
    with pytest.raises(TypeError):
        positive_sequence_conductance(20.2, "10")  # type: ignore


def test_get_transformer_clock():
    assert get_transformer_clock(30) == 1
    assert get_transformer_clock(45) == 1
    assert get_transformer_clock(360) == 12
    assert get_transformer_clock(105) == 3
    with pytest.raises(TypeError):
        get_transformer_clock("20.2")  # type: ignore
    with pytest.raises(TypeError):
        get_transformer_clock("105")  # type: ignore


def test_get_transformer_tap_size():
    assert get_transformer_tap_size(10, 9, 36, 0) == pytest.approx(3.60)
    assert get_transformer_tap_size(10, 9, 30, 1) == pytest.approx(2.70)
    assert get_transformer_tap_size(20, 10, 55.5, 1) == pytest.approx(5.55)
    assert get_transformer_tap_size(100, 101, 35, 0) == pytest.approx(35.00)
    with pytest.raises(TypeError):
        get_transformer_tap_size("10", "20", "35", 0)  # type: ignore
    with pytest.raises(ValueError):
        get_transformer_tap_size(10, 20, 35, 3)


def test_get_3wdgtransformer_tap_size():
    assert get_3wdgtransformer_tap_size(10, 9, 60, 36, 0) == pytest.approx(3.60)
    assert get_3wdgtransformer_tap_size(10, 54, 9, 30, 2) == pytest.approx(2.70)
    assert get_3wdgtransformer_tap_size(20, 13, 10, 55.5, 2) == pytest.approx(5.55)
    assert get_3wdgtransformer_tap_size(100, 50, 101, 35, 0) == pytest.approx(35.00)
    assert get_3wdgtransformer_tap_size(100, 50, 101, 35, 1) == pytest.approx(17.50)
    with pytest.raises(TypeError):
        get_3wdgtransformer_tap_size("10", "20", "30", "35", 0)  # type: ignore
    with pytest.raises(ValueError):
        get_transformer_tap_size(10, 20, 35, 4)


def test__split_string():
    assert _split_string("Yyn") == ("Y", "yn")
    assert _split_string("Dyn") == ("D", "yn")
    assert _split_string("YNyn") == ("YN", "yn")
    assert _split_string("YNd") == ("YN", "d")
    assert _split_string("Zyn") == ("Z", "yn")
    assert _split_string("ZNd") == ("ZN", "d")
    with pytest.raises(TypeError):
        _split_string(20)  # type: ignore
    with pytest.raises(TypeError):
        _split_string(32.51)  # type: ignore
    with pytest.raises(ValueError):
        _split_string("XNd")


def test_get_transformer_winding_from():
    assert get_transformer_winding_from("Yyn") == WindingType.wye
    assert get_transformer_winding_from("Dyn") == WindingType.delta
    assert get_transformer_winding_from("YNyn") == WindingType.wye_n
    assert get_transformer_winding_from("Zd") == WindingType.zigzag
    assert get_transformer_winding_from("ZNy") == WindingType.zigzag_n
    with pytest.raises(ValueError):
        get_transformer_winding_from("XNd")


def test_get_transformer_winding_to():
    assert get_transformer_winding_to("Yyn") == WindingType.wye_n
    assert get_transformer_winding_to("YNd") == WindingType.delta
    assert get_transformer_winding_to("Dy") == WindingType.wye
    assert get_transformer_winding_to("Dz") == WindingType.zigzag
    assert get_transformer_winding_to("YNzn") == WindingType.zigzag_n
    with pytest.raises(ValueError):
        get_transformer_winding_to("XNd")


def test__split_string_3wdg():
    assert _split_string_3wdg("Yynd") == ("Y", "yn", "d")
    assert _split_string_3wdg("Dyyn") == ("D", "y", "yn")
    assert _split_string_3wdg("Zynyn") == ("Z", "yn", "yn")
    assert _split_string_3wdg("Dynd") == ("D", "yn", "d")
    assert _split_string_3wdg("ZNyyn") == ("ZN", "y", "yn")
    assert _split_string_3wdg("YNdd") == ("YN", "d", "d")
    with pytest.raises(ValueError):
        _split_string_3wdg("XNd")
    with pytest.raises(TypeError):
        _split_string_3wdg(65.21)  # type: ignore


def test_get_3wdgtransformer_winding_1():
    assert get_3wdgtransformer_winding_1("Yynd") == WindingType.wye
    assert get_3wdgtransformer_winding_1("Dyny") == WindingType.delta
    assert get_3wdgtransformer_winding_1("YNzyn") == WindingType.wye_n
    assert get_3wdgtransformer_winding_1("Zdyn") == WindingType.zigzag
    assert get_3wdgtransformer_winding_1("ZNyd") == WindingType.zigzag_n
    with pytest.raises(TypeError):
        get_3wdgtransformer_winding_1(20)  # type: ignore
    with pytest.raises(TypeError):
        get_3wdgtransformer_winding_1(14.276)  # type: ignore
    with pytest.raises(ValueError):
        get_3wdgtransformer_winding_1("XNd")


def test_get_3wdgtransformer_winding_2():
    assert get_3wdgtransformer_winding_2("Yynd") == WindingType.wye_n
    assert get_3wdgtransformer_winding_2("Dyyn") == WindingType.wye
    assert get_3wdgtransformer_winding_2("YNzyn") == WindingType.zigzag
    assert get_3wdgtransformer_winding_2("Zdyn") == WindingType.delta
    assert get_3wdgtransformer_winding_2("ZNznd") == WindingType.zigzag_n
    with pytest.raises(TypeError):
        get_3wdgtransformer_winding_2(20)  # type: ignore
    with pytest.raises(TypeError):
        get_3wdgtransformer_winding_2(14.276)  # type: ignore
    with pytest.raises(ValueError):
        get_3wdgtransformer_winding_2("XNd")


def test_get_3wdgtransformer_winding_3():
    assert get_3wdgtransformer_winding_3("Yynd") == WindingType.delta
    assert get_3wdgtransformer_winding_3("Dyyn") == WindingType.wye_n
    assert get_3wdgtransformer_winding_3("YNzz") == WindingType.zigzag
    assert get_3wdgtransformer_winding_3("Zdy") == WindingType.wye
    assert get_3wdgtransformer_winding_3("ZNdzn") == WindingType.zigzag_n
    with pytest.raises(TypeError):
        get_3wdgtransformer_winding_3(20)  # type: ignore
    with pytest.raises(TypeError):
        get_3wdgtransformer_winding_3(14.276)  # type: ignore
    with pytest.raises(ValueError):
        get_3wdgtransformer_winding_3("XNd")


def test_get_3wdgtransformer_pk():
    assert get_3wdgtransformer_pk(14, 70, 20) == pytest.approx(280)
    assert get_3wdgtransformer_pk(10, 11.5, 54.35) == pytest.approx(115)
    assert get_3wdgtransformer_pk(200, 65.42, 30.5) == pytest.approx(6100)
    assert get_3wdgtransformer_pk(10, 15.1564, 91.23) == pytest.approx(151.564)
    with pytest.raises(TypeError):
        get_3wdgtransformer_pk("gsd", "rsgz", "dxgh")  # type: ignore
