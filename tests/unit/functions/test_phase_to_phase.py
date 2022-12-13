# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
import numpy as np
from power_grid_model.enum import WindingType
from pytest import approx, mark, raises

from power_grid_model_io.functions.phase_to_phase import (
    get_clock,
    get_winding_from,
    get_winding_to,
    power_wind_speed,
    reactive_power,
    reactive_power_to_susceptance,
    relative_no_load_current,
)


@mark.parametrize(
    ("i_0", "p_0", "s_nom", "u_nom", "expected"),
    [
        (float("nan"), float("nan"), float("nan"), float("nan"), float("nan")),
        (5.0, 1000.0, 100000.0, 400.0, 0.011547005383792516),
        (5.0, 2000.0, 100000.0, 400.0, 0.02),
    ],
)
def test_relative_no_load_current(i_0: float, p_0: float, s_nom: float, u_nom: float, expected: float):
    actual = relative_no_load_current(i_0, p_0, s_nom, u_nom)
    assert actual == approx(expected) or (np.isnan(actual) and np.isnan(expected))


def test_relative_no_load_current__exception():
    with raises(ValueError, match="can't be more than 100% .* 115.47%"):
        relative_no_load_current(500.0, 1000.0, 100000.0, 400.0)


@mark.parametrize(
    ("p", "cos_phi", "expected"),
    [
        (float("nan"), float("nan"), float("nan")),
        (1000.0, 0.9, 484.3221048378525),
    ],
)
def test_reactive_power(p: float, cos_phi: float, expected: float):
    actual = reactive_power(p, cos_phi)
    assert actual == approx(expected) or (np.isnan(actual) and np.isnan(expected))


@mark.parametrize(
    ("wind_speed", "expected"),
    [
        (0.0, 0.0),
        (1.5, 0.0),
        (3.0, 0.0),  # cut-in
        (8.5, 125000.0),
        (14.0, 1000000.0),  # nominal
        (19.5, 1000000.0),
        (25.0, 1000000.0),  # cutting-out
        (27.5, 500000.0),
        (30.0, 0.0),  # cut-out
        (100.0, 0.0),
    ],
)
def test_power_wind_speed(wind_speed, expected):
    assert power_wind_speed(1e6, wind_speed) == approx(expected)


@mark.parametrize(
    ("code", "winding_type"),
    [
        ("Yy1", WindingType.wye),
        ("Yyn2", WindingType.wye),
        ("Yd3", WindingType.wye),
        ("YNy4", WindingType.wye_n),
        ("YNyn5", WindingType.wye_n),
        ("YNd6", WindingType.wye_n),
        ("Dy7", WindingType.delta),
        ("Dyn8", WindingType.delta),
        ("Dd9", WindingType.delta),
        ("Zy2", WindingType.zigzag),
        ("Zy3", WindingType.zigzag),
        ("ZNy4", WindingType.zigzag_n),
        ("ZNy5", WindingType.zigzag_n),
    ],
)
def test_get_winding_from(code: str, winding_type: WindingType):
    assert get_winding_from(code) == winding_type


@mark.parametrize(
    ("code", "winding_type"),
    [
        ("Yy1", WindingType.wye),
        ("Yyn2", WindingType.wye),
        ("Yd3", WindingType.wye),
        ("YNy4", WindingType.wye),
        ("YNyn5", WindingType.wye),
        ("YNd6", WindingType.wye),
        ("Dy7", WindingType.delta),
        ("Dyn8", WindingType.delta),
        ("Dd9", WindingType.delta),
        ("Zy2", WindingType.zigzag),
        ("Zy3", WindingType.zigzag),
        ("ZNy4", WindingType.zigzag),
        ("ZNy5", WindingType.zigzag),
    ],
)
def test_get_winding_from__no_neutral_grounding(code: str, winding_type: WindingType):
    assert get_winding_from(code, False) == winding_type


def test_get_winding_from__exception():
    with raises(ValueError):
        get_winding_from("XNd11")


@mark.parametrize(
    ("code", "winding_type"),
    [
        ("Yy0", WindingType.wye),
        ("Yyn2", WindingType.wye_n),
        ("Yd3", WindingType.delta),
        ("YNy4", WindingType.wye),
        ("YNyn6", WindingType.wye_n),
        ("YNd7", WindingType.delta),
        ("Dy7", WindingType.wye),
        ("Dyn11", WindingType.wye_n),
        ("Dd8", WindingType.delta),
        ("Yz2", WindingType.zigzag),
        ("Yz4", WindingType.zigzag),
        ("Yzn4", WindingType.zigzag_n),
        ("Yzn6", WindingType.zigzag_n),
    ],
)
def test_get_winding_to(code: str, winding_type: WindingType):
    assert get_winding_to(code) == winding_type


@mark.parametrize(
    ("code", "winding_type"),
    [
        ("Yy0", WindingType.wye),
        ("Yyn2", WindingType.wye),
        ("Yd3", WindingType.delta),
        ("YNy4", WindingType.wye),
        ("YNyn4", WindingType.wye),
        ("YNd5", WindingType.delta),
        ("Dy7", WindingType.wye),
        ("Dyn9", WindingType.wye),
        ("Dd8", WindingType.delta),
        ("Yz2", WindingType.zigzag),
        ("Yz4", WindingType.zigzag),
        ("Yzn4", WindingType.zigzag),
        ("Yzn6", WindingType.zigzag),
    ],
)
def test_get_winding_to__no_neutral_grounding(code: str, winding_type: WindingType):
    assert get_winding_to(code, False) == winding_type


def test_get_winding_to__exception():
    with raises(ValueError):
        get_winding_to("YNx11")


@mark.parametrize(
    ("code", "clock"),
    [
        ("YNd0", 0),
        ("YNyn5", 5),
        ("YNd12", 12),
    ],
)
def test_get_clock(code: str, clock: int):
    assert get_clock(code) == clock


@mark.parametrize("code", ["YNd-1", "YNd13"])
def test_get_clock__exception(code):
    with raises(ValueError):
        get_clock(code)


@mark.parametrize(
    ("q_var", "u_nom", "expected"),
    [
        (float("nan"), float("nan"), float("nan")),
        (5000.0, 400, 0.03125),
    ],
)
def test_reactive_power_to_susceptance(q_var: float, u_nom: float, expected: float):
    actual = reactive_power_to_susceptance(q_var, u_nom)
    assert actual == approx(expected) or (np.isnan(actual) and np.isnan(expected))
