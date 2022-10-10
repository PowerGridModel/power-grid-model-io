import numpy as np
import pytest
from power_grid_model.data_types import BatchPythonDataset, SinglePythonDataset

from power_grid_model_io.converters.pgm_converter import PgmConverter
from power_grid_model_io.data_types import ExtraInfoLookup


@pytest.fixture
def converter():
    converter = PgmConverter()
    return converter


@pytest.fixture
def input_data():
    input_data = {
        "node": [
            {"id": 1, "u_rated": 400.0},
            {"id": 2, "u_rated": 400.0},
        ]
    }
    return input_data


@pytest.fixture
def batch_data():
    batch_data = [
        {"sym_load": [{"id": 3, "p_specified": 1.0}]},
        {"sym_load": [{"id": 3, "p_specified": 2.0}, {"id": 4, "p_specified": 3.0}]},
    ]
    return batch_data


def test_converter__parse_data(
    converter: PgmConverter, input_data: SinglePythonDataset, batch_data: BatchPythonDataset
):
    with pytest.raises(TypeError, match="Raw data should be either a list or a dictionary!"):
        converter._parse_data(data="str", data_type="input")  # type: ignore

    # test for input dataset
    pgm_data = converter._parse_data(data=input_data, data_type="input")
    assert len(pgm_data) == 1
    assert len(pgm_data["node"]) == 2
    assert [1, 2] in pgm_data["node"]["id"]
    assert [400.0, 400.0] in pgm_data["node"]["u_rated"]

    # test for batch dataset
    pgm_batch_data = converter._parse_data(data=batch_data, data_type="update")
    assert len(pgm_batch_data) == 1
    assert (pgm_batch_data["sym_load"]["indptr"] == np.array([0, 1, 3])).all()
    assert (pgm_batch_data["sym_load"]["data"]["id"] == [3, 3, 4]).all()
    assert (pgm_batch_data["sym_load"]["data"]["p_specified"] == [1.0, 2.0, 3.0]).all()

    # TODO include extra_info


def test_converter__parse_dataset(converter: PgmConverter, input_data: SinglePythonDataset):
    pgm_data = converter._parse_dataset(data=input_data, data_type="input")

    assert len(pgm_data) == 1
    assert len(pgm_data["node"]) == 2
    assert [1, 2] in pgm_data["node"]["id"]
    assert [400.0, 400.0] in pgm_data["node"]["u_rated"]

    # TODO include extra_info


def test_converter__parse_component(converter: PgmConverter, input_data: SinglePythonDataset):
    objects = list(input_data.values())
    component = "node"
    extra_node = {"id": 3, "u_rated": 400.0, "some_extra_info": 1}
    objects[0].append(extra_node)
    extra_info: ExtraInfoLookup = {}

    node_array = converter._parse_component(
        objects=objects[0], component=component, data_type="input", extra_info=extra_info
    )
    assert (len(node_array)) == 3
    assert [1, 2, 3] in node_array["id"]
    assert [400.0, 400.0, 400.0] in node_array["u_rated"]
    assert extra_info == {3: {"some_extra_info": 1}}

    node_with_wrong_attr_val = {"id": 3, "u_rated": "fault"}
    objects[0].append(node_with_wrong_attr_val)
    with pytest.raises(
        ValueError, match="Invalid 'u_rated' value for node input data: could not convert string to float: 'fault'"
    ):
        converter._parse_component(objects=objects[0], component=component, data_type="input")
