# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path
from typing import Callable
from unittest.mock import ANY, MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest
from power_grid_model import WindingType

from power_grid_model_io.converters.pandapower_converter import PandaPowerConverter

from ...utils import MockDf, MockFn

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "pandapower"


def _generate_ids(*args, **kwargs):
    return MockFn("_generate_ids", *args, **kwargs)


def _get_pgm_ids(*args, **kwargs):
    return MockFn("_get_pgm_ids", *args, **kwargs)


def _get_pp_attr(*args, **kwargs):
    return MockFn("_get_pp_attr", *args, **kwargs)


def get_switch_states(*args, **kwargs):
    return MockFn("get_switch_states", *args, **kwargs)


def get_trafo3w_switch_states(*args, **kwargs):
    return MockFn("get_trafo3w_switch_states", *args, **kwargs)


def get_trafo_winding_types(*args, **kwargs):
    return MockFn("get_trafo_winding_types", *args, **kwargs)


def get_trafo3w_winding_types(*args, **kwargs):
    return MockFn("get_trafo3w_winding_types", *args, **kwargs)


def _get_tap_size(*args, **kwargs):
    return MockFn("_get_tap_size", *args, **kwargs)


def np_array(*args, **kwargs):
    return MockFn("np.array", *args, **kwargs)


@pytest.fixture
def converter() -> PandaPowerConverter:
    converter = PandaPowerConverter()
    converter._generate_ids = MagicMock(side_effect=_generate_ids)  # type: ignore
    converter._get_pgm_ids = MagicMock(side_effect=_get_pgm_ids)  # type: ignore
    converter._get_pp_attr = MagicMock(side_effect=_get_pp_attr)  # type: ignore
    converter.get_switch_states = MagicMock(side_effect=get_switch_states)  # type: ignore
    converter.get_trafo_winding_types = MagicMock(side_effect=get_trafo_winding_types)  # type: ignore
    converter._get_tap_size = MagicMock(side_effect=_get_tap_size)  # type: ignore
    converter.get_trafo_winding_types = MagicMock(side_effect=get_trafo_winding_types)  # type: ignore
    converter.get_trafo3w_switch_states = MagicMock(side_effect=get_trafo3w_switch_states)  # type: ignore
    converter.get_trafo3w_winding_types = MagicMock(side_effect=get_trafo3w_winding_types)  # type: ignore
    return converter


@pytest.fixture
def two_pp_objs() -> MockDf:
    return MockDf(2)


@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._fill_extra_info")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._create_input_data")
def test_parse_data(create_input_data_mock: MagicMock, fill_extra_info_mock: MagicMock):
    # Arrange
    converter = PandaPowerConverter()

    def create_input_data():
        converter.pgm_input_data = {"node": np.array([])}

    create_input_data_mock.side_effect = create_input_data

    # Act
    result = converter._parse_data(data={"bus": pd.DataFrame()}, data_type="input", extra_info=None)

    # Assert
    create_input_data_mock.assert_called_once_with()
    fill_extra_info_mock.assert_not_called()
    assert len(converter.pp_input_data) == 1 and "bus" in converter.pp_input_data
    assert len(converter.pgm_input_data) == 1 and "node" in converter.pgm_input_data
    assert len(result) == 1 and "node" in result


@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._fill_extra_info")
@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._create_input_data")
def test_parse_data__extra_info(create_input_data_mock: MagicMock, fill_extra_info_mock: MagicMock):
    # Arrange
    converter = PandaPowerConverter()

    extra_info = MagicMock("extra_info")

    # Act
    converter._parse_data(data={}, data_type="input", extra_info=extra_info)

    # Assert
    create_input_data_mock.assert_called_once_with()
    fill_extra_info_mock.assert_called_once_with(extra_info=extra_info)


def test_parse_data__update_data():
    # Arrange
    converter = PandaPowerConverter()

    # Act/Assert
    with pytest.raises(ValueError):
        converter._parse_data(data={}, data_type="update", extra_info=None)


def test_fill_extra_info():
    # Arrange
    converter = PandaPowerConverter()
    converter.idx_lookup[("bus", None)] = pd.Series([101, 102, 103], index=[0, 1, 2])
    converter.idx_lookup[("load", "const_current")] = pd.Series([201, 202, 203], index=[3, 4, 5])

    # Act
    extra_info = {}
    converter._fill_extra_info(extra_info=extra_info)

    # Assert
    assert len(extra_info) == 6
    assert extra_info[0] == {"id_reference": {"table": "bus", "index": 101}}
    assert extra_info[1] == {"id_reference": {"table": "bus", "index": 102}}
    assert extra_info[2] == {"id_reference": {"table": "bus", "index": 103}}
    assert extra_info[3] == {"id_reference": {"table": "load", "name": "const_current", "index": 201}}
    assert extra_info[4] == {"id_reference": {"table": "load", "name": "const_current", "index": 202}}
    assert extra_info[5] == {"id_reference": {"table": "load", "name": "const_current", "index": 203}}


def test_serialize_data():
    # Arrange
    converter = PandaPowerConverter()

    # Act/Assert
    with pytest.raises(NotImplementedError):
        converter._serialize_data(data={}, extra_info=None)


def test_create_input_data():
    # Arrange
    converter = MagicMock()

    # Act
    PandaPowerConverter._create_input_data(self=converter)  # type: ignore

    # Assert
    assert len(converter.method_calls) == 16
    converter._create_pgm_input_nodes.assert_called_once_with()
    converter._create_pgm_input_lines.assert_called_once_with()
    converter._create_pgm_input_sources.assert_called_once_with()
    converter._create_pgm_input_sym_loads.assert_called_once_with()
    converter._create_pgm_input_asym_loads.assert_called_once_with()
    converter._create_pgm_input_shunts.assert_called_once_with()
    converter._create_pgm_input_transformers.assert_called_once_with()
    converter._create_pgm_input_sym_gens.assert_called_once_with()
    converter._create_pgm_input_asym_gens.assert_called_once_with()
    converter._create_pgm_input_three_winding_transformers.assert_called_once_with()
    converter._create_pgm_input_links.assert_called_once_with()
    converter._create_pgm_input_storage.assert_called_once_with()
    converter._create_pgm_input_impedance.assert_called_once_with()
    converter._create_pgm_input_ward.assert_called_once_with()
    converter._create_pgm_input_xward.assert_called_once_with()
    converter._create_pgm_input_motor.assert_called_once_with()


@pytest.mark.parametrize(
    ("create_fn", "table"),
    [
        (PandaPowerConverter._create_pgm_input_nodes, "bus"),
        (PandaPowerConverter._create_pgm_input_lines, "line"),
        (PandaPowerConverter._create_pgm_input_sources, "ext_grid"),
        (PandaPowerConverter._create_pgm_input_shunts, "shunt"),
        (PandaPowerConverter._create_pgm_input_sym_gens, "sgen"),
        (PandaPowerConverter._create_pgm_input_sym_loads, "load"),
        (PandaPowerConverter._create_pgm_input_transformers, "trafo"),
        (PandaPowerConverter._create_pgm_input_three_winding_transformers, "trafo3w"),
        (PandaPowerConverter._create_pgm_input_links, "switch"),
    ],
)
@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_object__empty(
    mock_init_array: MagicMock, create_fn: Callable[[PandaPowerConverter], None], table: str
):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data[table] = pd.DataFrame()  # type: ignore

    # Act
    create_fn(converter)

    # Assert
    mock_init_array.assert_not_called()


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_nodes(mock_init_array: MagicMock, two_pp_objs: MockDf, converter):
    # Arrange
    converter.pp_input_data["bus"] = two_pp_objs

    # Act
    converter._create_pgm_input_nodes()

    # Assert

    # administration
    converter._generate_ids.assert_called_once_with("bus", two_pp_objs.index)

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="node", shape=2)

    # retrieval
    converter._get_pp_attr.assert_any_call("bus", "vn_kv")
    assert len(converter._get_pp_attr.call_args_list) == 1

    # assignment
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("bus", two_pp_objs.index))
    pgm.assert_any_call("u_rated", _get_pp_attr("bus", "vn_kv") * 1e3)
    assert len(pgm.call_args_list) == 2

    # result
    assert converter.pgm_input_data["node"] == mock_init_array.return_value


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_lines(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["line"] = two_pp_objs

    # Act
    converter._create_pgm_input_lines()

    # Assert

    # administration
    converter.get_switch_states.assert_called_once_with("line")
    converter._generate_ids.assert_called_once_with("line", two_pp_objs.index)
    converter._get_pgm_ids.assert_any_call("bus", two_pp_objs["from_bus"])
    converter._get_pgm_ids.assert_any_call("bus", two_pp_objs["to_bus"])

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="line", shape=2)

    # retrieval
    converter._get_pp_attr.assert_any_call("line", "in_service", True)
    converter._get_pp_attr.assert_any_call("line", "length_km")
    converter._get_pp_attr.assert_any_call("line", "parallel", 1)
    converter._get_pp_attr.assert_any_call("line", "r_ohm_per_km")
    converter._get_pp_attr.assert_any_call("line", "x_ohm_per_km")
    converter._get_pp_attr.assert_any_call("line", "c_nf_per_km")
    converter._get_pp_attr.assert_any_call("line", "g_us_per_km", 0)
    converter._get_pp_attr.assert_any_call("line", "max_i_ka")
    converter._get_pp_attr.assert_any_call("line", "df", 1)
    assert len(converter._get_pp_attr.call_args_list) == 9

    # assignment
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", _generate_ids("line", two_pp_objs.index))
    pgm.assert_any_call("from_node", _get_pgm_ids("bus", two_pp_objs["from_bus"]))
    pgm.assert_any_call("from_status", _get_pp_attr("line", "in_service", True) & get_switch_states("line")["from"])
    pgm.assert_any_call("to_node", _get_pgm_ids("bus", two_pp_objs["to_bus"]))
    pgm.assert_any_call("to_status", _get_pp_attr("line", "in_service", True) & get_switch_states("line")["to"])
    pgm.assert_any_call(
        "r1",
        _get_pp_attr("line", "r_ohm_per_km")
        * (_get_pp_attr("line", "length_km") / _get_pp_attr("line", "parallel", 1)),
    )
    pgm.assert_any_call(
        "x1",
        _get_pp_attr("line", "x_ohm_per_km")
        * (_get_pp_attr("line", "length_km") / _get_pp_attr("line", "parallel", 1)),
    )
    pgm.assert_any_call(
        "c1",
        _get_pp_attr("line", "c_nf_per_km")
        * _get_pp_attr("line", "length_km")
        * _get_pp_attr("line", "parallel", 1)
        * 1e-9,
    )
    pgm.assert_any_call(
        "tan1", _get_pp_attr("line", "g_us_per_km", 0) / _get_pp_attr("line", "c_nf_per_km") / (np.pi / 10)
    )
    pgm.assert_any_call(
        "i_n",
        _get_pp_attr("line", "max_i_ka") * 1e3 * _get_pp_attr("line", "df", 1) * _get_pp_attr("line", "parallel", 1),
    )
    assert len(pgm.call_args_list) == 10

    # result
    assert converter.pgm_input_data["line"] == mock_init_array.return_value


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_sources(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["ext_grid"] = two_pp_objs

    # Act
    converter._create_pgm_input_sources()

    # Assert

    # administration: TODO

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="source", shape=2)

    # retrieval:
    converter._get_pp_attr.assert_any_call("ext_grid", "bus")
    converter._get_pp_attr.assert_any_call("ext_grid", "vm_pu")
    converter._get_pp_attr.assert_any_call("ext_grid", "va_degree")
    converter._get_pp_attr.assert_any_call("ext_grid", "s_sc_max_mva")
    converter._get_pp_attr.assert_any_call("ext_grid", "rx_max")
    converter._get_pp_attr.assert_any_call("ext_grid", "in_service", True)
    assert len(converter._get_pp_attr.call_args_list) == 6

    # assignment:
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", ANY)
    pgm.assert_any_call("node", ANY)
    pgm.assert_any_call("status", ANY)
    pgm.assert_any_call("u_ref", ANY)
    pgm.assert_any_call("u_ref_angle", ANY)
    pgm.assert_any_call("sk", ANY)
    pgm.assert_any_call("rx_ratio", ANY)
    assert len(pgm.call_args_list) == 7

    # result
    assert converter.pgm_input_data["source"] == mock_init_array.return_value


@pytest.mark.xfail(reason="Not implemented")
@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_sym_loads(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data[...] = two_pp_objs

    # Act
    converter._create_pgm_input_sym_loads()

    # Assert

    # administration: TODO

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type="sym_load", shape=6)

    # retrieval: TODO
    converter._get_pp_attr.assert_any_call(..., ...)
    assert len(converter._get_pp_attr.call_args_list) == 0

    # assignment: TODO

    # result
    assert converter.pgm_input_data[...] == mock_init_array.return_value


@pytest.mark.xfail(reason="Not implemented")
@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_asym_loads(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data[...] = two_pp_objs

    # Act
    converter._create_pgm_input_asym_loads()

    # Assert

    # administration: TODO

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type=..., shape=2)

    # retrieval: TODO
    converter._get_pp_attr.assert_any_call("asymmetric_load", ...)
    assert len(converter._get_pp_attr.call_args_list) == 0

    # assignment: TODO
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", ANY)
    pgm.assert_any_call(..., ANY)
    assert len(pgm.call_args_list) == 0

    # result
    assert converter.pgm_input_data[...] == mock_init_array.return_value


@pytest.mark.xfail(reason="Not implemented")
@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_shunts(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data[...] = two_pp_objs

    # Act
    converter._create_pgm_input_shunts()

    # Assert

    # administration: TODO

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type=..., shape=2)

    # retrieval: TODO
    converter._get_pp_attr.assert_any_call(..., ...)
    assert len(converter._get_pp_attr.call_args_list) == 0

    # assignment: TODO
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", ANY)
    pgm.assert_any_call(..., ANY)

    assert len(pgm.call_args_list) == 0

    # result
    assert converter.pgm_input_data[...] == mock_init_array.return_value


@pytest.mark.xfail(reason="Not implemented")
@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_transformers(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data[...] = two_pp_objs

    # Act
    converter._create_pgm_input_transformers()
    # Assert

    # administration: TODO

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type=..., shape=2)

    # retrieval: TODO
    converter._get_pp_attr.assert_any_call(..., ...)
    assert len(converter._get_pp_attr.call_args_list) == 0

    # assignment: TODO
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", ANY)
    pgm.assert_any_call(..., ANY)

    assert len(pgm.call_args_list) == 0

    # result
    assert converter.pgm_input_data[...] == mock_init_array.return_value


@pytest.mark.xfail(reason="Not implemented")
@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_sym_gens(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data[...] = two_pp_objs

    # Act
    converter._create_pgm_input_sym_gens()

    # Assert

    # administration: TODO

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type=..., shape=2)

    # retrieval: TODO
    converter._get_pp_attr.assert_any_call(..., ...)
    assert len(converter._get_pp_attr.call_args_list) == 0

    # assignment: TODO
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", ANY)
    pgm.assert_any_call(..., ANY)
    assert len(pgm.call_args_list) == 0

    # result
    assert converter.pgm_input_data[...] == mock_init_array.return_value


@pytest.mark.xfail(reason="Not implemented")
@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_asym_gens(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data[...] = two_pp_objs

    # Act
    converter._create_pgm_input_asym_gens()

    # Assert

    # administration: TODO

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type=..., shape=2)

    # retrieval: TODO
    converter._get_pp_attr.assert_any_call(..., ...)
    assert len(converter._get_pp_attr.call_args_list) == 0

    # assignment: TODO
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", ANY)
    pgm.assert_any_call(..., ANY)
    assert len(pgm.call_args_list) == 0

    # result
    assert converter.pgm_input_data[...] == mock_init_array.return_value


@pytest.mark.xfail(reason="Not implemented")
@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_three_winding_transformers(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data[...] = two_pp_objs

    # Act
    converter._create_pgm_input_three_winding_transformers()

    # Assert

    # administration: TODO

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type=..., shape=2)

    # retrieval: TODO
    converter._get_pp_attr.assert_any_call(..., ...)
    assert len(converter._get_pp_attr.call_args_list) == 0

    # assignment: TODO
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", ANY)
    pgm.assert_any_call(..., ANY)
    assert len(pgm.call_args_list) == 0

    # result
    assert converter.pgm_input_data[...] == mock_init_array.return_value


@pytest.mark.xfail(reason="Not implemented")
@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_links(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data[...] = two_pp_objs

    # Act
    converter._create_pgm_input_sources()

    # Assert

    # administration: TODO

    # initialization
    mock_init_array.assert_called_once_with(data_type="input", component_type=..., shape=2)

    # retrieval: TODO
    converter._get_pp_attr.assert_any_call(..., ...)
    assert len(converter._get_pp_attr.call_args_list) == 0

    # assignment: TODO
    pgm: MagicMock = mock_init_array.return_value.__setitem__
    pgm.assert_any_call("id", ANY)
    pgm.assert_any_call(..., ANY)
    assert len(pgm.call_args_list) == 0

    # result
    assert converter.pgm_input_data[...] == mock_init_array.return_value


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_storage(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["storage"] = two_pp_objs

    # Act / Assert
    with pytest.raises(NotImplementedError, match=r"Storage.*not implemented"):
        converter._create_pgm_input_storage()

    # initialization
    mock_init_array.assert_not_called()


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_impedance(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["impedance"] = two_pp_objs

    # Act / Assert
    with pytest.raises(NotImplementedError, match=r"Impedance.*not implemented"):
        converter._create_pgm_input_impedance()

    # initialization
    mock_init_array.assert_not_called()


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_ward(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["ward"] = two_pp_objs

    # Act / Assert
    with pytest.raises(NotImplementedError, match=r"Ward.*not implemented"):
        converter._create_pgm_input_ward()

    # initialization
    mock_init_array.assert_not_called()


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_xward(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["xward"] = two_pp_objs

    # Act / Assert
    with pytest.raises(NotImplementedError, match=r"Extended Ward.*not implemented"):
        converter._create_pgm_input_xward()

    # initialization
    mock_init_array.assert_not_called()


@patch("power_grid_model_io.converters.pandapower_converter.initialize_array")
def test_create_pgm_input_motor(mock_init_array: MagicMock, two_pp_objs, converter):
    # Arrange
    converter.pp_input_data["motor"] = two_pp_objs

    # Act / Assert
    with pytest.raises(NotImplementedError, match=r"Motor.*not implemented"):
        converter._create_pgm_input_motor()

    # initialization
    mock_init_array.assert_not_called()


def test_get_pgm_ids():
    # Arrange
    converter = PandaPowerConverter()
    converter.idx = {
        ("bus", None): pd.Series([10, 11, 12], index=[0, 1, 2]),
        ("load", "const_current"): pd.Series([13, 14], index=[3, 4]),
    }

    # Act
    bus_ids = converter._get_pgm_ids(pp_table="bus", pp_idx=pd.Series([2, 1]))
    load_ids = converter._get_pgm_ids(pp_table="load", name="const_current", pp_idx=pd.Series([3]))
    all_bus_ids = converter._get_pgm_ids(pp_table="bus")

    # Assert
    pd.testing.assert_series_equal(bus_ids, pd.Series([12, 11], index=[2, 1]))
    pd.testing.assert_series_equal(load_ids, pd.Series([13], index=[3]))
    pd.testing.assert_series_equal(all_bus_ids, pd.Series([10, 11, 12], index=[0, 1, 2]))


def test_get_pgm_ids__key_error():
    # Arrange
    converter = PandaPowerConverter()

    # Act / Assert
    with pytest.raises(KeyError, match=r"index.*bus"):
        converter._get_pgm_ids(pp_table="bus")


def test_get_tap_size():
    # Arrange
    pp_trafo = pd.DataFrame(
        [["hv", 5.0, 10.5, 0.4], ["lv", 10.0, 10.5, 0.4], ["lv", np.nan, 10.5, 0.4]],
        columns=["tap_side", "tap_step_percent", "vn_hv_kv", "vn_lv_kv"],
    )
    expected_tap_size = np.array([525.0, 40.0, np.nan], dtype=np.float64)

    # Act
    actual_tap_size = PandaPowerConverter._get_tap_size(pp_trafo)

    # Assert
    np.testing.assert_array_equal(actual_tap_size, expected_tap_size)


def test_get_transformer_tap_side():
    # Arrange
    pp_trafo_tap_side = pd.Series(["hv", "lv", "lv", "hv"])
    expected_tap_side = np.array([0, 1, 1, 0], dtype=np.int8)

    # Act
    actual_tap_side = PandaPowerConverter._get_transformer_tap_side(pp_trafo_tap_side)

    # Assert
    np.testing.assert_array_equal(actual_tap_side, expected_tap_side)


def test_get_3wtransformer_tap_side():
    # Arrange
    pp_trafo3w_tap_side = pd.Series(["hv", "mv", "lv", "mv", "lv", "hv", "lv"])
    expected_tap_side = np.array([0, 1, 2, 1, 2, 0, 2], dtype=np.int8)

    # Act
    actual_tap_side = PandaPowerConverter._get_3wtransformer_tap_side(pp_trafo3w_tap_side)

    # Assert
    np.testing.assert_array_equal(actual_tap_side, expected_tap_side)


def test_get_3wtransformer_tap_size():
    # Arrange
    pp_trafo = pd.DataFrame(
        [["hv", 62.0, 10.5, 400.0, 32.1], ["lv", 62.0, 10.5, 400.0, 32.1], ["mv", 62.0, 10.5, 400.0, 32.1]],
        columns=["tap_side", "tap_step_percent", "vn_hv_kv", "vn_mv_kv", "vn_lv_kv"],
    )
    expected_tap_size = np.array([6510.0, 19902.0, 248000.0], dtype=np.float64)

    # Act
    actual_tap_size = PandaPowerConverter._get_3wtransformer_tap_size(pp_trafo)

    # Assert
    np.testing.assert_array_equal(actual_tap_size, expected_tap_size)


@patch("power_grid_model_io.converters.pandapower_converter.get_winding")
def test_get_trafo_winding_types__vector_group(mock_get_winding: MagicMock):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {
        "trafo": pd.DataFrame([(1, "Dyn"), (2, "YNd"), (3, "Dyn")], columns=["id", "vector_group"])
    }
    mock_get_winding.side_effect = [WindingType.delta, WindingType.wye_n, WindingType.wye_n, WindingType.delta]
    expected = pd.DataFrame([(2, 1), (1, 2), (2, 1)], columns=["winding_from", "winding_to"])

    # Act
    actual = converter.get_trafo_winding_types()

    # Assert
    pd.testing.assert_frame_equal(actual, expected)
    assert len(mock_get_winding.call_args_list) == 4
    assert mock_get_winding.call_args_list[0] == call("D")
    assert mock_get_winding.call_args_list[1] == call("yn")
    assert mock_get_winding.call_args_list[2] == call("YN")
    assert mock_get_winding.call_args_list[3] == call("d")


@patch("power_grid_model_io.converters.pandapower_converter.get_winding")
def test_get_trafo_winding_types__std_types(mock_get_winding: MagicMock):
    # Arrange
    std_types = {"trafo": {"std_trafo_1": {"vector_group": "YNd"}, "std_trafo_2": {"vector_group": "Dyn"}}}
    converter = PandaPowerConverter(std_types=std_types)
    converter.pp_input_data = {
        "trafo": pd.DataFrame([(1, "std_trafo_2"), (2, "std_trafo_1"), (3, "std_trafo_2")], columns=["id", "std_type"])
    }
    mock_get_winding.side_effect = [WindingType.delta, WindingType.wye_n, WindingType.wye_n, WindingType.delta]
    expected = pd.DataFrame([(2, 1), (1, 2), (2, 1)], columns=["winding_from", "winding_to"])

    # Act
    actual = converter.get_trafo_winding_types()

    # Assert
    pd.testing.assert_frame_equal(actual, expected)
    assert len(mock_get_winding.call_args_list) == 4
    assert mock_get_winding.call_args_list[0] == call("D")
    assert mock_get_winding.call_args_list[1] == call("yn")
    assert mock_get_winding.call_args_list[2] == call("YN")
    assert mock_get_winding.call_args_list[3] == call("d")


@patch("power_grid_model_io.converters.pandapower_converter.get_winding")
def test_get_trafo3w_winding_types__vector_group(mock_get_winding: MagicMock):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {
        "trafo3w": pd.DataFrame([(1, "Dynz"), (2, "YNdy"), (3, "Dyny")], columns=["id", "vector_group"])
    }
    mock_get_winding.side_effect = [
        WindingType.delta,
        WindingType.wye_n,
        WindingType.zigzag,
        WindingType.wye_n,
        WindingType.delta,
        WindingType.wye,
        WindingType.delta,
        WindingType.wye_n,
        WindingType.wye,
    ]

    expected = pd.DataFrame([[2, 1, 3], [1, 2, 0], [2, 1, 0]], columns=["winding_1", "winding_2", "winding_3"])

    # Act
    actual = converter.get_trafo3w_winding_types()

    # Assert
    pd.testing.assert_frame_equal(actual, expected)
    assert len(mock_get_winding.call_args_list) == 9
    assert mock_get_winding.call_args_list[0] == call("D")
    assert mock_get_winding.call_args_list[1] == call("yn")
    assert mock_get_winding.call_args_list[2] == call("z")
    assert mock_get_winding.call_args_list[3] == call("YN")
    assert mock_get_winding.call_args_list[4] == call("d")
    assert mock_get_winding.call_args_list[5] == call("y")
    assert mock_get_winding.call_args_list[6] == call("D")
    assert mock_get_winding.call_args_list[7] == call("yn")
    assert mock_get_winding.call_args_list[8] == call("y")


@patch("power_grid_model_io.converters.pandapower_converter.get_winding")
def test_get_trafo3w_winding_types__std_types(mock_get_winding: MagicMock):
    # Arrange
    std_types = {"trafo3w": {"std_trafo3w_1": {"vector_group": "Dynz"}, "std_trafo3w_2": {"vector_group": "YNdy"}}}
    converter = PandaPowerConverter(std_types=std_types)
    converter.pp_input_data = {
        "trafo3w": pd.DataFrame(
            [(1, "std_trafo3w_2"), (2, "std_trafo3w_1"), (3, "std_trafo3w_2")], columns=["id", "std_type"]
        )
    }
    mock_get_winding.side_effect = [
        WindingType.wye_n,
        WindingType.delta,
        WindingType.wye,
        WindingType.delta,
        WindingType.wye_n,
        WindingType.zigzag,
        WindingType.wye_n,
        WindingType.delta,
        WindingType.wye,
    ]
    expected = pd.DataFrame([[1, 2, 0], [2, 1, 3], [1, 2, 0]], columns=["winding_1", "winding_2", "winding_3"])

    # Act
    actual = converter.get_trafo3w_winding_types()

    # Assert
    pd.testing.assert_frame_equal(actual, expected)
    assert len(mock_get_winding.call_args_list) == 6
    assert mock_get_winding.call_args_list[0] == call("YN")
    assert mock_get_winding.call_args_list[1] == call("d")
    assert mock_get_winding.call_args_list[2] == call("y")
    assert mock_get_winding.call_args_list[3] == call("D")
    assert mock_get_winding.call_args_list[4] == call("yn")
    assert mock_get_winding.call_args_list[5] == call("z")


def test_get_winding_types__value_error():
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {"trafo": pd.DataFrame([(1, "ADyn")], columns=["id", "vector_group"])}

    # Act / Assert
    with pytest.raises(ValueError):
        converter.get_trafo_winding_types()


def test_get_trafo3w_winding_types__value_error():
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {"trafo3w": pd.DataFrame([(1, "ADyndrr")], columns=["id", "vector_group"])}

    # Act / Assert
    with pytest.raises(ValueError):
        converter.get_trafo3w_winding_types()


def test_get_individual_switch_states():
    # Arrange
    pp_trafo = pd.DataFrame(
        columns=["index", "hv_bus"],
        data=[[1, 101], [2, 102], [3, 103]],
    )
    pp_switches = pd.DataFrame(
        columns=["element", "bus", "closed"],
        data=[[1, 101, False], [2, 202, False], [3, 103, True]],
    )

    expected_state = pd.Series([False, True, True], dtype=bool)

    # Act
    actual_state = PandaPowerConverter.get_individual_switch_states(pp_trafo, pp_switches, "hv_bus")

    # Assert
    np.testing.assert_array_equal(actual_state, expected_state)


def test_get_id():
    # Arrange
    converter = PandaPowerConverter()
    converter.idx = {("line", None): pd.Series([21, 345, 0, 3, 15], index=[0, 1, 2, 3, 4])}

    # Act
    actual_id = converter.get_id("line", 1)

    # Assert
    np.testing.assert_array_equal(actual_id, 345)


@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter.get_individual_switch_states")
def test_get_switch_states_lines(mock_get_individual_switch_states: MagicMock):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {
        "line": pd.DataFrame(columns=["from_bus", "to_bus"], data=[[101, 102]], index=[1]),
        "switch": pd.DataFrame(
            columns=["bus", "et", "element", "closed"],
            data=[[101, "l", 1, False], [102, "x", 1, False]],
            index=[1001, 1002],
        ),
    }
    mock_get_individual_switch_states.side_effect = [
        pd.Series([False], index=[1], dtype=bool, name="closed"),
        pd.Series([True], index=[1], dtype=bool, name="closed"),
    ]
    expected = pd.DataFrame(columns=["from", "to"], index=[1], data=[[False, True]])

    # Act
    actual = converter.get_switch_states("line")

    # Assert
    pd.testing.assert_frame_equal(actual, expected)

    assert len(mock_get_individual_switch_states.call_args_list) == 2

    assert mock_get_individual_switch_states.call_args_list[0] == call(ANY, ANY, "from_bus")
    pd.testing.assert_frame_equal(
        mock_get_individual_switch_states.call_args_list[0].args[0],
        pd.DataFrame(columns=["index", "from_bus"], data=[[1, 101]], index=[1]),
    )
    pd.testing.assert_frame_equal(
        mock_get_individual_switch_states.call_args_list[0].args[1],
        pd.DataFrame(columns=["element", "bus", "closed"], data=[[1, 101, False]], index=[1001]),
    )

    assert mock_get_individual_switch_states.call_args_list[1] == call(ANY, ANY, "to_bus")
    pd.testing.assert_frame_equal(
        mock_get_individual_switch_states.call_args_list[1].args[0],
        pd.DataFrame(columns=["index", "to_bus"], data=[[1, 102]], index=[1]),
    )
    pd.testing.assert_frame_equal(
        mock_get_individual_switch_states.call_args_list[1].args[1],
        pd.DataFrame(columns=["element", "bus", "closed"], data=[[1, 101, False]], index=[1001]),
    )


@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter.get_individual_switch_states")
def test_get_switch_states_trafos(mock_get_individual_switch_states: MagicMock):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {
        "trafo": pd.DataFrame([[2, 32, 31]], columns=["index", "hv_bus", "lv_bus"]),
        "switch": pd.DataFrame(
            [[101, 32, "t", 2, True], [321, 31, "t", 2, False]],
            columns=["index", "bus", "et", "element", "closed"],
        ),
    }
    mock_get_individual_switch_states.side_effect = [
        pd.Series([False], index=[2], dtype=bool, name="closed"),
        pd.Series([True], index=[2], dtype=bool, name="closed"),
    ]

    expected = pd.DataFrame(columns=["from", "to"], index=[2], data=[[False, True]])

    # Act
    actual = converter.get_switch_states("trafo")

    # Assert
    pd.testing.assert_frame_equal(actual, expected)
    assert len(mock_get_individual_switch_states.call_args_list) == 2


def test_get_switch_states__exception():
    # Arrange
    converter = PandaPowerConverter()

    # Act / Assert
    with pytest.raises(KeyError, match=r"link"):
        converter.get_switch_states("link")


@patch("power_grid_model_io.converters.pandapower_converter.PandaPowerConverter.get_individual_switch_states")
def test_get_trafo3w_switch_states(mock_get_individual_switch_states: MagicMock):
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {
        "trafo3w": pd.DataFrame([[2, 32, 31, 315]], columns=["index", "hv_bus", "mv_bus", "lv_bus"]),
        "switch": pd.DataFrame(
            [[101, 315, "t3", 2, False], [321, 32, "t3", 2, False]],
            columns=["index", "bus", "et", "element", "closed"],
        ),
    }
    mock_get_individual_switch_states.side_effect = [False, True, False]

    expected = pd.DataFrame(data=([False], [True], [False]))

    # Act
    actual = converter.get_trafo3w_switch_states(converter.pp_input_data["trafo3w"])

    # Assert
    pd.testing.assert_frame_equal(actual, expected)
    assert len(mock_get_individual_switch_states.call_args_list) == 3


def test_lookup_id():
    # Arrange
    converter = PandaPowerConverter()
    converter.idx_lookup = {
        ("line", None): pd.Series([0, 1, 2, 3, 4], index=[21, 345, 0, 3, 15]),
        ("load", "const_current"): pd.Series([5, 6, 7, 8, 9], index=[543, 14, 34, 48, 4]),
    }

    expected_line = {"table": "line", "index": 4}
    expected_load = {"table": "load", "name": "const_current", "index": 8}

    # Act
    actual_line = converter.lookup_id(15)
    actual_load = converter.lookup_id(48)

    # Assert
    np.testing.assert_array_equal(actual_line, expected_line)
    np.testing.assert_array_equal(actual_load, expected_load)


def test_lookup_id__value_error():
    # Arrange
    converter = PandaPowerConverter()
    converter.idx_lookup = {("line", None): pd.Series([0, 1, 2, 3, 4], index=[21, 345, 0, 3, 15])}

    # Act / Assert
    with pytest.raises(KeyError):
        converter.lookup_id(5)


def test_get_pp_attr_attribute_exists():
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {
        "trafo3w": pd.DataFrame([[2, 32, 31, 315]], columns=["index", "hv_bus", "mv_bus", "lv_bus"])
    }
    expected = np.array(32)

    # Act
    actual = converter._get_pp_attr("trafo3w", "hv_bus")

    # Assert
    np.testing.assert_array_equal(actual, expected)


def test_get_pp_attr_attribute_doesnt_exist():
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {"trafo3w": pd.DataFrame([[2, 31, 315]], columns=["index", "mv_bus", "lv_bus"])}

    # Act / Assert
    with pytest.raises(KeyError):
        converter._get_pp_attr("trafo3w", "hv_bus")


def test_get_pp_attr_use_default():
    # Arrange
    converter = PandaPowerConverter()
    converter.pp_input_data = {"trafo3w": pd.DataFrame([[2, 31, 315]], columns=["index", "mv_bus", "lv_bus"])}
    expected = np.array(625)

    # Act
    actual = converter._get_pp_attr("trafo3w", "hv_bus", 625)

    # Assert
    np.testing.assert_array_equal(actual, expected)


def test_get_pp_attr_from_std():
    # Arrange
    converter = PandaPowerConverter()
    converter._std_types = {"trafo3w": {"std_trafo3w_1": {"hv_bus": 964}}}
    converter.pp_input_data = {
        "trafo3w": pd.DataFrame([[2, 31, 315, "std_trafo3w_1"]], columns=["index", "mv_bus", "lv_bus", "std_type"])
    }

    expected = np.array(964)

    # Act
    actual = converter._get_pp_attr("trafo3w", "hv_bus")

    # Assert
    np.testing.assert_array_equal(actual, expected)


def test_get_pp_attr_default_after_checking_std():
    # Arrange
    converter = PandaPowerConverter()
    converter._std_types = {"trafo3w": {"std_trafo3w_1": {"lv_bus": 23}}}
    converter.pp_input_data = {
        "trafo3w": pd.DataFrame([[2, 31, 315, "std_trafo3w_1"]], columns=["index", "mv_bus", "lv_bus", "std_type"])
    }

    expected = np.array(964)

    # Act
    actual = converter._get_pp_attr("trafo3w", "hv_bus", 964)

    # Assert
    np.testing.assert_array_equal(actual, expected)


def test_get_pp_attr_error_after_checking_std():
    # Arrange
    converter = PandaPowerConverter()
    converter._std_types = {"trafo3w": {"std_trafo3w_1": {"lv_bus": 23}}}
    converter.pp_input_data = {
        "trafo3w": pd.DataFrame([[2, 31, 315, "std_trafo3w_1"]], columns=["index", "mv_bus", "lv_bus", "std_type"])
    }

    # Act/Assert
    with pytest.raises(KeyError):
        converter._get_pp_attr("trafo3w", "hv_bus")
