# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import json
import warnings
from collections.abc import Callable
from functools import lru_cache
from importlib import metadata
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandapower as pp
import pandas as pd
import pytest
from packaging import version
from pandapower import runpp_3ph
from pandapower.networks import example_simple
from power_grid_model import (
    AttributeType as AT,
    Branch3Side,
    BranchSide,
    ComponentType as CT,
    DatasetType,
    PowerGridModel,
    WindingType,
)
from power_grid_model.data_types import SingleDataset
from power_grid_model.validation import assert_valid_input_data

from power_grid_model_io._enum import PandapowerAttribute as _PpAttr, PandapowerTable as _PpTable
from power_grid_model_io.converters import PandaPowerConverter
from power_grid_model_io.converters.pandapower_converter import (
    PP_COMPATIBILITY_VERSION_3_4_0,
    PP_CONVERSION_VERSION,
)
from power_grid_model_io.data_types import ExtraInfo
from power_grid_model_io.utils.json import JsonEncoder
from tests.data.pandapower.pp_validation import pp_net, pp_net_3ph_minimal_trafo, pp_net_pv_node_3
from tests.validation.utils import (
    compare_extra_info,
    component_attributes,
    component_objects,
    load_json_single_dataset,
    select_values,
)

pytestmark = pytest.mark.pandapower

PANDAPOWER_DATA_DIR = Path(__file__).parents[2] / "data" / "pandapower"
VALIDATION_FILE = PANDAPOWER_DATA_DIR / "pgm_input_data.json"
VALIDATION_FILE_ZERO_SEQ = PANDAPOWER_DATA_DIR / "pgm_input_data_trafo_zero_seq.json"
PV_VALIDATION_FILE = PANDAPOWER_DATA_DIR / "pv-node" / "pv-node3" / "pgm_input.json"

type PandaPowerNet = pp.pandapowerNet

mag0_multiplier = 1.0 if PP_CONVERSION_VERSION < PP_COMPATIBILITY_VERSION_3_4_0 else 100.0


@lru_cache
def load_and_convert_pp_data() -> tuple[SingleDataset, ExtraInfo]:
    """
    Load and convert the pandapower validation network
    """
    net = pp_net()
    pp_converter = PandaPowerConverter()
    data, extra_info = pp_converter.load_input_data(net)
    return data, extra_info


@lru_cache
def load_validation_data(file: Path = VALIDATION_FILE) -> tuple[SingleDataset, ExtraInfo]:
    """
    Load the validation data from the json file
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        data, extra_info = load_json_single_dataset(file, data_type=DatasetType.input)

    return data, extra_info


@pytest.fixture
def input_data() -> tuple[SingleDataset, SingleDataset]:
    """
    Load the pandapower network and the json file, and return the input_data
    """
    actual, _ = load_and_convert_pp_data()
    expected, _ = load_validation_data()
    return actual, expected


@pytest.fixture
def extra_info() -> tuple[ExtraInfo, ExtraInfo]:
    """
    Load the pandapower network and the json file, and return the extra_info
    """
    _, actual = load_and_convert_pp_data()
    _, expected = load_validation_data()
    return actual, expected


def test_input_data(input_data: tuple[SingleDataset, SingleDataset]):
    """
    Unit test to preload the expected and actual data
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # Arrange
        actual, expected = input_data

        # Assert
        assert len(expected) <= len(actual)


@pytest.mark.parametrize(
    ("component", "attribute"), list(component_attributes(VALIDATION_FILE, data_type=DatasetType.input))
)
def test_attributes(input_data: tuple[SingleDataset, SingleDataset], component: CT, attribute: AT):
    """
    For each attribute, check if the actual values are consistent with the expected values
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # Arrange
        actual_data, expected_data = input_data

        # Act
        actual_values, expected_values = select_values(actual_data, expected_data, component, attribute)

        # Assert
        if isinstance(actual_values, pd.Series) and isinstance(expected_values, pd.Series):
            pd.testing.assert_series_equal(actual_values, expected_values)
        else:
            pd.testing.assert_frame_equal(actual_values, expected_values)


@pytest.mark.parametrize(
    ("component", "obj_ids"),
    [pytest.param(component, objects, id=component) for component, objects in component_objects(VALIDATION_FILE)],
)
def test_extra_info(extra_info: tuple[ExtraInfo, ExtraInfo], component: CT, obj_ids: list[int]):
    """
    For each object, check if the actual extra info is consistent with the expected extra info
    """
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # Arrange
        actual, expected = extra_info

        # Assert
        errors = compare_extra_info(actual=actual, expected=expected, component=component, obj_ids=obj_ids)

        # Raise a value error, containing all the errors at once
        if errors:
            raise ValueError("\n" + "\n".join(errors))


def test_extra_info__serializable(extra_info):
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # Arrange
        actual, _expected = extra_info

        # Assert
        json.dumps(actual, cls=JsonEncoder)  # expect no exception


def test_pgm_input_lines__cnf_zero():
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        pp_network = pp_net_3ph_minimal_trafo()
        pp_converter = PandaPowerConverter()
        pp_network.line.c_nf_per_km = 0
        data, _ = pp_converter.load_input_data(pp_network)
        np.testing.assert_array_equal(data[CT.line][AT.tan1], 0)

        pp_network.line.c_nf_per_km = 0.001
        pp_network.line.c0_nf_per_km = 0
        data, _ = pp_converter.load_input_data(pp_network)
        np.testing.assert_array_equal(data[CT.line][AT.tan0], 0)


@pytest.mark.filterwarnings("error")
def test_simple_example():
    pp_net = example_simple()
    pp_converter = PandaPowerConverter()
    _, _ = pp_converter.load_input_data(pp_net)


@pytest.mark.filterwarnings("error")
def test_simple_example_with_strings():
    pp_net = example_simple()

    for table, attrs in {
        _PpTable.bus: [_PpAttr.type],
        _PpTable.load: [_PpAttr.type],
        _PpTable.asymmetric_load: [_PpAttr.type],
        _PpTable.sgen: [_PpAttr.type],
        _PpTable.gen: [_PpAttr.type],
        _PpTable.line: [_PpAttr.type],
        _PpTable.trafo: [_PpAttr.vector_group, _PpAttr.tap_side],
        _PpTable.trafo3w: [_PpAttr.tap_side],
        _PpTable.switch: [_PpAttr.type],
    }.items():
        for attr in attrs:
            pp_net[table][attr] = pp_net[table][attr].astype(pd.StringDtype())

    pp_converter = PandaPowerConverter()
    _, _ = pp_converter.load_input_data(pp_net)


@pytest.mark.filterwarnings("error")
def test_simple_example_with_na():
    pp_net = example_simple()

    for table, attrs in {
        _PpTable.line: [_PpAttr.type],
        _PpTable.trafo: [_PpAttr.vector_group, _PpAttr.tap_side],
        _PpTable.switch: [_PpAttr.type],
    }.items():
        for attr in attrs:
            pp_net[table][attr] = pp_net[table][attr].astype(pd.StringDtype())
            pp_net[table].loc[0, attr] = pd.NA

    pp_converter = PandaPowerConverter()
    _, _ = pp_converter.load_input_data(pp_net)


def test_trafo_zero_seq_params_conversion():
    net = pp_net_3ph_minimal_trafo()
    net.trafo.vector_group = "YNyn"
    net.trafo.mag0_percent = 1e3 * mag0_multiplier
    net.trafo.mag0_rx = 0.1
    net.trafo.shift_degree = 0
    converter = PandaPowerConverter()
    actual_data, _ = converter.load_input_data(net)
    expected_data, _ = load_validation_data(VALIDATION_FILE_ZERO_SEQ)
    np.testing.assert_allclose(
        actual_data[CT.transformer][AT.i0_zero_sequence],
        expected_data[CT.transformer][AT.i0_zero_sequence],
        rtol=1e-3,
    )
    np.testing.assert_allclose(
        actual_data[CT.transformer][AT.p0_zero_sequence],
        expected_data[CT.transformer][AT.p0_zero_sequence],
        rtol=1e-3,
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {_PpAttr.mag0_percent: 1e2},
        {_PpAttr.mag0_percent: 1e3},
        {_PpAttr.mag0_percent: 1e6},
        {_PpAttr.mag0_percent: 1e20},
    ],
)
def test_trafo_zero_seq_params_calculation(kwargs):
    net = pp_net_3ph_minimal_trafo()
    net.trafo.vector_group = "YNyn"
    net.trafo.mag0_percent = kwargs[_PpAttr.mag0_percent] * mag0_multiplier
    net.trafo.mag0_rx = 0.1
    net.trafo.shift_degree = 0
    converter = PandaPowerConverter()
    data, _ = converter.load_input_data(net)
    assert_valid_input_data(data)
    pgm_net = PowerGridModel(data)
    output = pgm_net.calculate_power_flow(symmetric=False)
    runpp_3ph(net)
    assert np.allclose(
        output[CT.transformer][AT.p_from][0] / 1e6,
        net.res_trafo_3ph.loc[0, [_PpAttr.p_a_hv_mw, _PpAttr.p_b_hv_mw, _PpAttr.p_c_hv_mw]].values,
        rtol=1e-3,
    )


def test_trafo_negative_tap_step():
    pp_net = example_simple()
    pp_net[_PpTable.trafo][_PpAttr.tap_step_percent] *= -1
    pp_converter = PandaPowerConverter()
    pgm_data, _ = pp_converter.load_input_data(pp_net)
    assert pgm_data["transformer"][0]["tap_size"] > 0
    assert pgm_data["transformer"][0]["tap_max"] < pgm_data["transformer"][0]["tap_min"]

    assert_valid_input_data(pgm_data)


def test_create_input_gen__voltage_regultor():
    pp_net = pp_net_pv_node_3()
    pp_converter = PandaPowerConverter()
    actual_data, _ = pp_converter.load_input_data(pp_net)

    expected_data, _ = load_validation_data(PV_VALIDATION_FILE)

    for component, attribute in component_attributes(PV_VALIDATION_FILE, DatasetType.input):
        # Act
        actual_values, expected_values = select_values(actual_data, expected_data, component, attribute)

        # Assert
        if isinstance(actual_values, pd.Series) and isinstance(expected_values, pd.Series):
            pd.testing.assert_series_equal(actual_values, expected_values)
        else:
            pd.testing.assert_frame_equal(actual_values, expected_values)


@pytest.mark.parametrize("kwargs", [{_PpAttr.r0x0_max: 0.5, _PpAttr.rx_max: 4}, {_PpAttr.x0x_max: 0.6}])
def test_create_pgm_input_sources__zero_sequence(kwargs) -> None:
    pp_net: PandaPowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=1.0)
    pp.create_ext_grid(pp_net, 0, **kwargs)

    converter = PandaPowerConverter()
    converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}
    converter.idx = {(_PpTable.bus, None): pd.Series([0], index=[0])}

    with patch("power_grid_model_io.converters.pandapower_converter.logger") as mock_logger:
        converter._create_pgm_input_sources()
        mock_logger.warning.assert_called_once()


def test_create_pgm_input_sym_loads__delta() -> None:
    # Arrange
    pp_net: PandaPowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=0.0)
    pp.create_load(pp_net, 0, 0, type="delta")

    converter = PandaPowerConverter()
    converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}

    # Act/Assert
    with pytest.raises(
        NotImplementedError, match=r"Delta loads are not implemented, only wye loads are supported in PGM."
    ):
        converter._create_pgm_input_sym_loads()


def test_create_pgm_input_asym_loads__delta() -> None:
    # Arrange
    pp_net: PandaPowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=0.0)
    pp.create_asymmetric_load(pp_net, 0, type="delta")

    converter = PandaPowerConverter()
    converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}

    # Act/Assert
    with pytest.raises(
        NotImplementedError, match=r"Delta loads are not implemented, only wye loads are supported in PGM."
    ):
        converter._create_pgm_input_asym_loads()


def test_create_pgm_input_transformers__tap_dependent_impedance() -> None:
    # Arrange
    pp_net: PandaPowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=0.0)
    args = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    if version.Version(metadata.version("pandapower")) >= version.Version("3"):
        with pytest.deprecated_call():
            pp.create_transformer_from_parameters(pp_net, *args, tap_dependent_impedance=True)
    else:
        pp.create_transformer_from_parameters(pp_net, *args, tap_dependent_impedance=True)

        converter = PandaPowerConverter()
        converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}

        # Act/Assert
        with pytest.raises(RuntimeError, match="not supported"):
            converter._create_pgm_input_transformers()


@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter.get_switch_states",
    new=MagicMock(return_value=pd.DataFrame({"from": [True], "to": [True]})),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter.get_trafo_winding_types",
    new=MagicMock(
        return_value=pd.DataFrame(
            {
                "winding_from": [0, 0, 0, 0, 0, 0, WindingType.delta, WindingType.wye_n, 0, 0],
                "winding_to": [0, 0, 0, 0, 0, 0, WindingType.wye_n, WindingType.wye_n, 0, 0],
            }
        )
    ),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._generate_ids",
    new=MagicMock(return_value=np.arange(1)),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._get_pgm_ids",
    new=MagicMock(return_value=pd.Series([0])),
)
def test_create_pgm_input_transformers__default() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # Arrange
        pp_net: PandaPowerNet = pp.create_empty_network()
        pp.create_bus(net=pp_net, vn_kv=0.0)
        args = [0, 0, 1, 0, 0, 0, 0, 0, 0]
        pp.create_transformer_from_parameters(
            pp_net, *args, tap_neutral=12.0, tap_pos=34.0, tap_step_percent=1, tap_side="hv"
        )
        pp.create_transformer_from_parameters(
            pp_net, *args, tap_neutral=12.0, tap_pos=34.0, tap_step_percent=1, tap_side="lv"
        )
        pp.create_transformer_from_parameters(
            pp_net, *args, tap_neutral=12.0, tap_pos=34.0, tap_step_percent=1, tap_side=None
        )
        tap_pos_trafo = pp.create_transformer_from_parameters(
            pp_net, *args, tap_neutral=12.0, tap_size=1, tap_side="hv"
        )
        pp_net[_PpTable.trafo].loc[tap_pos_trafo, "tap_pos"] = np.nan
        pp.create_transformer_from_parameters(pp_net, *args, tap_neutral=np.nan, tap_pos=34.0, tap_side="hv")
        pp.create_transformer_from_parameters(
            pp_net, *args, tap_neutral=12, tap_step_percent=np.nan, tap_pos=34.0, tap_side="hv"
        )
        pp.create_transformer_from_parameters(pp_net, *args, vector_group=None, shift_degree=30)
        pp.create_transformer_from_parameters(pp_net, *args, vector_group=None, shift_degree=60)
        pp.create_transformer_from_parameters(pp_net, *args, vector_group=None, shift_degree=59)
        pp.create_transformer_from_parameters(pp_net, *args, vector_group=None, shift_degree=61)

        converter = PandaPowerConverter()
        converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}

        # Act
        converter._create_pgm_input_transformers()
        result = converter.pgm_input_data[CT.transformer]

        # Assert
        assert result[0][AT.tap_side] == BranchSide.from_side.value
        assert result[1][AT.tap_side] == BranchSide.to_side.value
        assert result[2][AT.tap_side] == BranchSide.from_side.value
        assert result[3][AT.tap_side] == BranchSide.from_side.value
        assert result[4][AT.tap_side] == BranchSide.from_side.value
        assert result[0][AT.tap_pos] == 34.0 != result[0][AT.tap_nom]
        assert result[1][AT.tap_pos] == 34.0 != result[1][AT.tap_nom]
        assert result[2][AT.tap_pos] == 0.0 == result[2][AT.tap_nom]
        assert result[3][AT.tap_pos] == 0.0 == result[3][AT.tap_nom]
        assert result[4][AT.tap_pos] == 0.0 == result[4][AT.tap_nom]
        assert result[5][AT.tap_size] == 0.0

        assert result[6][AT.winding_from] == WindingType.delta
        assert result[6][AT.winding_to] == WindingType.wye_n
        assert result[7][AT.winding_from] == WindingType.wye_n
        assert result[7][AT.winding_to] == WindingType.wye_n

        assert result[8][AT.clock] == 2
        assert result[9][AT.clock] == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {_PpAttr.pfe_kw: 2},
        {_PpAttr.vk0_percent: 2},
        {_PpAttr.vkr0_percent: 1},
        {_PpAttr.si0_hv_partial: 0.3},
    ],
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter.get_switch_states",
    new=MagicMock(return_value=pd.DataFrame({"from": [True], "to": [True]})),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter.get_trafo_winding_types",
    new=MagicMock(return_value=pd.DataFrame({"winding_from": [0], "winding_to": [0]})),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._generate_ids",
    new=MagicMock(return_value=np.arange(1)),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._get_pgm_ids",
    new=MagicMock(return_value=pd.Series([0])),
)
def test_create_pgm_input_transformers__warnings(kwargs) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # Arrange
        pp_net: PandaPowerNet = pp.create_empty_network()
        pp.create_bus(net=pp_net, vn_kv=0.0)
        args = [0, 0, 1, 0, 0, 0, 0, 0, 0]
        if _PpAttr.pfe_kw in kwargs:
            args[-2] = kwargs[_PpAttr.pfe_kw]
            kwargs = {}
        pp.create_transformer_from_parameters(pp_net, *args, **kwargs)

        converter = PandaPowerConverter()
        converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}

        with patch("power_grid_model_io.converters.pandapower_converter.logger") as mock_logger:
            converter._create_pgm_input_transformers()
            mock_logger.warning.assert_called_once()


@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter.get_trafo3w_switch_states",
    new=MagicMock(return_value=pd.DataFrame({"side_1": [True], "side_2": [True], "side_3": [True]})),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter.get_trafo3w_winding_types",
    new=MagicMock(
        return_value=pd.DataFrame(
            {
                "winding_1": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    WindingType.wye_n,
                    WindingType.wye_n,
                    WindingType.wye_n,
                    WindingType.wye_n,
                    0,
                    0,
                ],
                "winding_2": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    WindingType.delta,
                    WindingType.wye_n,
                    WindingType.wye_n,
                    WindingType.delta,
                    0,
                    0,
                ],
                "winding_3": [
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    WindingType.delta,
                    WindingType.wye_n,
                    WindingType.delta,
                    WindingType.wye_n,
                    0,
                    0,
                ],
            }
        )
    ),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._generate_ids",
    new=MagicMock(return_value=np.arange(1)),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._get_pgm_ids",
    new=MagicMock(return_value=pd.Series([0])),
)
def test_create_pgm_input_transformers3w__default() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # Arrange
        pp_net: PandaPowerNet = pp.create_empty_network()
        pp.create_bus(net=pp_net, vn_kv=0.0)
        args = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        pp.create_transformer3w_from_parameters(
            pp_net, *args, tap_neutral=12.0, tap_pos=34.0, tap_step_percent=1, tap_side="hv"
        )
        pp.create_transformer3w_from_parameters(
            pp_net, *args, tap_neutral=12.0, tap_pos=34.0, tap_step_percent=1, tap_side="mv"
        )
        pp.create_transformer3w_from_parameters(
            pp_net, *args, tap_neutral=12.0, tap_pos=34.0, tap_step_percent=1, tap_side="lv"
        )
        pp.create_transformer3w_from_parameters(
            pp_net, *args, tap_neutral=12.0, tap_pos=34.0, tap_step_percent=1, tap_side=None
        )
        pp.create_transformer3w_from_parameters(
            pp_net, *args, tap_neutral=np.nan, tap_pos=34.0, tap_step_percent=1, tap_side="hv"
        )
        nan_trafo = pp.create_transformer3w_from_parameters(
            pp_net, *args, tap_neutral=12.0, tap_step_percent=1, tap_pos=np.nan, tap_side="hv"
        )
        pp_net[_PpTable.trafo3w].loc[nan_trafo, _PpAttr.tap_pos] = np.nan
        pp.create_transformer3w_from_parameters(
            pp_net, *args, tap_neutral=12.0, tap_pos=34.0, tap_step_percent=np.nan, tap_side="hv"
        )
        pp.create_transformer3w_from_parameters(
            pp_net,
            *args,
            vector_group=None,
            shift_mv_degree=30,
            shift_lv_degree=30,
        )
        pp.create_transformer3w_from_parameters(
            pp_net,
            *args,
            vector_group=None,
            shift_mv_degree=60,
            shift_lv_degree=60,
        )
        pp.create_transformer3w_from_parameters(
            pp_net,
            *args,
            vector_group=None,
            shift_mv_degree=60,
            shift_lv_degree=30,
        )
        pp.create_transformer3w_from_parameters(
            pp_net,
            *args,
            vector_group=None,
            shift_mv_degree=30,
            shift_lv_degree=60,
        )
        pp.create_transformer3w_from_parameters(
            pp_net,
            *args,
            vector_group=None,
            shift_mv_degree=58,
            shift_lv_degree=62,
        )
        pp.create_transformer3w_from_parameters(
            pp_net,
            *args,
            vector_group=None,
            shift_mv_degree=29,
            shift_lv_degree=31,
        )

        converter = PandaPowerConverter()
        converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}

        # Act
        converter._create_pgm_input_three_winding_transformers()
        result = converter.pgm_input_data[CT.three_winding_transformer]

        # Assert
        assert result[0][AT.tap_side] == Branch3Side.side_1.value
        assert result[1][AT.tap_side] == Branch3Side.side_2.value
        assert result[2][AT.tap_side] == Branch3Side.side_3.value
        assert result[3][AT.tap_side] == Branch3Side.side_1.value
        assert result[4][AT.tap_side] == Branch3Side.side_1.value
        assert result[5][AT.tap_side] == Branch3Side.side_1.value
        assert result[0][AT.tap_pos] == 34.0 != result[0][AT.tap_nom]
        assert result[1][AT.tap_pos] == 34.0 != result[1][AT.tap_nom]
        assert result[2][AT.tap_pos] == 34.0 != result[2][AT.tap_nom]
        assert result[3][AT.tap_pos] == 0 == result[3][AT.tap_nom]
        assert result[4][AT.tap_pos] == 0 == result[4][AT.tap_nom]
        assert result[5][AT.tap_pos] == 0 == result[5][AT.tap_nom]
        assert result[6][AT.tap_size] == 0

        # Default yndd for odd clocks
        assert result[7][AT.winding_1] == WindingType.wye_n
        assert result[7][AT.winding_2] == WindingType.delta
        assert result[7][AT.winding_3] == WindingType.delta
        # Default ynynyn for even clocks
        assert result[8][AT.winding_1] == WindingType.wye_n
        assert result[8][AT.winding_2] == WindingType.wye_n
        assert result[8][AT.winding_3] == WindingType.wye_n
        # Default ynynd for clock_12 even clock_13 odd
        assert result[9][AT.winding_1] == WindingType.wye_n
        assert result[9][AT.winding_2] == WindingType.wye_n
        assert result[9][AT.winding_3] == WindingType.delta
        # Default yndyn for clock_12 odd clock_13 even
        assert result[10][AT.winding_1] == WindingType.wye_n
        assert result[10][AT.winding_2] == WindingType.delta
        assert result[10][AT.winding_3] == WindingType.wye_n

        assert result[11][AT.clock_12] == result[11][AT.clock_13] == 2
        assert result[12][AT.clock_12] == result[12][AT.clock_13] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {_PpAttr.pfe_kw: 2},
        {_PpAttr.vk0_hv_percent: 1},
        {_PpAttr.vkr0_hv_percent: 2},
        {_PpAttr.vk0_mv_percent: 3},
        {_PpAttr.vkr0_mv_percent: 4},
        {_PpAttr.vk0_lv_percent: 5},
        {_PpAttr.vkr0_lv_percent: 6},
    ],
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter.get_trafo3w_switch_states",
    new=MagicMock(return_value=pd.DataFrame({"side_1": [True], "side_2": [True], "side_3": [True]})),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter.get_trafo3w_winding_types",
    new=MagicMock(return_value=pd.DataFrame({"winding_1": [0], "winding_2": [0], "winding_3": [0]})),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._generate_ids",
    new=MagicMock(return_value=np.arange(1)),
)
@patch(
    "power_grid_model_io.converters.pandapower_converter.PandaPowerConverter._get_pgm_ids",
    new=MagicMock(return_value=pd.Series([0])),
)
def test_create_pgm_input_transformers3w__warnings(kwargs) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("error")

        # Arrange
        pp_net: PandaPowerNet = pp.create_empty_network()
        pp.create_bus(net=pp_net, vn_kv=0.0)
        args = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
        if _PpAttr.pfe_kw in kwargs:
            args[-2] = kwargs[_PpAttr.pfe_kw]
            kwargs = {}
        pp.create_transformer3w_from_parameters(pp_net, *args, **kwargs)

        converter = PandaPowerConverter()
        converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}

        # Act
        with patch("power_grid_model_io.converters.pandapower_converter.logger") as mock_logger:
            converter._create_pgm_input_three_winding_transformers()
            mock_logger.warning.assert_called_once()


def test_create_pgm_input_three_winding_transformers__tap_at_star_point() -> None:
    # Arrange
    pp_net: PandaPowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=0.0)
    args = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    pp.create_transformer3w_from_parameters(pp_net, *args, tap_at_star_point=True)

    converter = PandaPowerConverter()
    converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}

    # Act/Assert
    with pytest.raises(RuntimeError, match="not supported"):
        converter._create_pgm_input_three_winding_transformers()


def test_create_pgm_input_three_winding_transformers__tap_dependent_impedance() -> None:
    # Arrange
    pp_net: PandaPowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=0.0)
    args = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    if version.Version(metadata.version("pandapower")) >= version.Version("3"):
        with pytest.deprecated_call():
            pp.create_transformer3w_from_parameters(pp_net, *args, tap_dependent_impedance=True)
    else:
        pp.create_transformer3w_from_parameters(pp_net, *args, tap_dependent_impedance=True)

        converter = PandaPowerConverter()
        converter.pp_input_data = {k: v for k, v in pp_net.items() if isinstance(v, pd.DataFrame)}

        # Act/Assert
        with pytest.raises(RuntimeError, match="not supported"):
            converter._create_pgm_input_three_winding_transformers()


def test_create_pgm_input_wards__existing_loads() -> None:
    converter = PandaPowerConverter()
    # Arrange

    pp_net: PandaPowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=0.0)
    pp.create_load(pp_net, 0, 0)
    pp.create_ward(pp_net, 0, 0, 0, 0, 0)

    converter.pp_input_data = pp_net

    # Act
    converter._create_pgm_input_nodes()
    converter._create_pgm_input_sym_loads()
    converter._create_pgm_input_wards()

    # assert
    assert len(converter.pgm_input_data[CT.sym_load]) == 5


def test_create_pgm_input_motors__existing_loads() -> None:
    converter = PandaPowerConverter()
    # Arrange

    pp_net: PandaPowerNet = pp.create_empty_network()
    pp.create_bus(net=pp_net, vn_kv=0.0)
    pp.create_load(pp_net, 0, 0)
    pp.create_motor(pp_net, 0, 0, 0)

    converter.pp_input_data = pp_net

    # Act
    converter._create_pgm_input_nodes()
    converter._create_pgm_input_sym_loads()
    converter._create_pgm_input_motors()

    # assert
    assert len(converter.pgm_input_data[CT.sym_load]) == 4


@pytest.mark.parametrize(
    "create_fn",
    [
        PandaPowerConverter._create_pgm_input_sources,
        PandaPowerConverter._create_pgm_input_shunts,
        PandaPowerConverter._create_pgm_input_lines,
        PandaPowerConverter._create_pgm_input_sgens,
        PandaPowerConverter._create_pgm_input_gens,
        PandaPowerConverter._create_pgm_input_sym_loads,
        PandaPowerConverter._create_pgm_input_asym_gens,
        PandaPowerConverter._create_pgm_input_asym_loads,
        PandaPowerConverter._create_pgm_input_impedances,
        PandaPowerConverter._create_pgm_input_links,
        PandaPowerConverter._create_pgm_input_motors,
        PandaPowerConverter._create_pgm_input_nodes,
        PandaPowerConverter._create_pgm_input_storages,
        PandaPowerConverter._create_pgm_input_three_winding_transformers,
        PandaPowerConverter._create_pgm_input_transformers,
        PandaPowerConverter._create_pgm_input_wards,
        PandaPowerConverter._create_pgm_input_xwards,
        PandaPowerConverter._create_pgm_input_dclines,
    ],
)
def test_create_pp_input_object__empty(create_fn: Callable[[PandaPowerConverter], None]):
    # Arrange: No table
    converter = PandaPowerConverter()
    converter.pp_input_data = pp.create_empty_network()

    # Act / Assert
    with patch("power_grid_model_io.converters.pandapower_converter.initialize_array") as mock_init_array:
        create_fn(converter)
        mock_init_array.assert_not_called()
