# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pytest
from power_grid_model import initialize_array
from power_grid_model.data_types import BatchDataset, SingleDataset
from structlog.testing import capture_logs

from power_grid_model_io.converters.pgm_json_converter import PgmJsonConverter
from power_grid_model_io.data_types import ExtraInfoLookup


@pytest.fixture
def converter():
    converter = PgmJsonConverter()
    return converter


@pytest.fixture
def structured_input_data():
    input_data = {
        "node": [
            {"id": 1, "u_rated": 400.0},
            {"id": 2, "u_rated": 400.0, "some_extra_info": 2.1},
        ]
    }
    return input_data


@pytest.fixture
def structured_batch_data():
    batch_data = [
        {"sym_load": [{"id": 3, "p_specified": 1.0}]},
        {"sym_load": [{"id": 3, "p_specified": 2.0}, {"id": 4, "p_specified": 3.0}]},
    ]
    return batch_data


@pytest.fixture
def pgm_input_data():
    node = initialize_array("input", "node", 2)
    node["id"] = [1, 2]
    return {"node": node}


@pytest.fixture
def pgm_batch_data():
    line = initialize_array("update", "line", (3, 2))
    return {"line": line}


@pytest.fixture
def pgm_sparse_batch_data():
    array = {"indptr": np.array([0, 1, 2, 3]), "data": np.array(["dummy", "data", "array"])}
    return {"component_name": array}


def test_converter__parse_data(converter: PgmJsonConverter, structured_input_data, structured_batch_data):
    with pytest.raises(TypeError, match="Raw data should be either a list or a dictionary!"):
        converter._parse_data(data="str", data_type="input", extra_info=None)  # type: ignore

    # test for input dataset
    extra_info: ExtraInfoLookup = {}
    pgm_data = converter._parse_data(data=structured_input_data, data_type="input", extra_info=extra_info)
    assert len(pgm_data) == 1
    assert len(pgm_data["node"]) == 2
    assert [1, 2] in pgm_data["node"]["id"]
    assert [400.0, 400.0] in pgm_data["node"]["u_rated"]
    assert extra_info == {2: {"some_extra_info": 2.1}}

    # test for batch dataset
    pgm_batch_data = converter._parse_data(data=structured_batch_data, data_type="update", extra_info=None)
    assert len(pgm_batch_data) == 1
    assert (pgm_batch_data["sym_load"]["indptr"] == np.array([0, 1, 3])).all()
    assert (pgm_batch_data["sym_load"]["data"]["id"] == [3, 3, 4]).all()
    assert (pgm_batch_data["sym_load"]["data"]["p_specified"] == [1.0, 2.0, 3.0]).all()


def test_converter__parse_dataset(converter: PgmJsonConverter, structured_input_data):
    extra_info: ExtraInfoLookup = {}
    pgm_data = converter._parse_dataset(data=structured_input_data, data_type="input", extra_info=extra_info)

    assert len(pgm_data) == 1
    assert len(pgm_data["node"]) == 2
    assert [1, 2] in pgm_data["node"]["id"]
    assert [400.0, 400.0] in pgm_data["node"]["u_rated"]
    assert extra_info == {2: {"some_extra_info": 2.1}}


def test_converter__parse_component(converter: PgmJsonConverter, structured_input_data):
    objects = list(structured_input_data.values())
    component = "node"
    extra_info: ExtraInfoLookup = {}

    node_array = converter._parse_component(
        objects=objects[0], component=component, data_type="input", extra_info=extra_info
    )
    assert (len(node_array)) == 2
    assert [1, 2] in node_array["id"]
    assert [400.0, 400.0] in node_array["u_rated"]
    assert extra_info == {2: {"some_extra_info": 2.1}}

    node_with_wrong_attr_val = {"id": 3, "u_rated": "fault"}
    objects[0].append(node_with_wrong_attr_val)  # type: ignore
    with pytest.raises(
        ValueError, match="Invalid 'u_rated' value for node input data: could not convert string to float: 'fault'"
    ):
        converter._parse_component(objects=objects[0], component=component, data_type="input", extra_info=None)


def test_converter__serialize_data(
    converter: PgmJsonConverter, pgm_input_data: SingleDataset, pgm_batch_data: BatchDataset
):
    structured_single_data = converter._serialize_data(data=pgm_input_data, extra_info=None)
    assert structured_single_data == {"node": [{"id": 1}, {"id": 2}]}
    with capture_logs() as cap_log:
        structured_batch_data = converter._serialize_data(data=pgm_batch_data, extra_info={})
    assert structured_batch_data == [{"line": [{}, {}]}, {"line": [{}, {}]}, {"line": [{}, {}]}]
    assert cap_log[0]["event"] == "Extra info is not supported for batch data export"
    assert cap_log[0]["log_level"] == "warning"


def test_converter__is_batch(
    converter: PgmJsonConverter,
    pgm_input_data: SingleDataset,
    pgm_batch_data: BatchDataset,
    pgm_sparse_batch_data: BatchDataset,
):
    # Single dataset
    assert not converter._is_batch(pgm_input_data)
    # Dense batch dataset
    assert converter._is_batch(pgm_batch_data)
    # Sparse batch dataset
    assert converter._is_batch(pgm_sparse_batch_data)
    # Wrong dataset with both single and batch data
    combined_input_batch = pgm_input_data | pgm_batch_data
    with pytest.raises(ValueError, match=r"Mixed non-batch data with batch data \(line\)."):
        converter._is_batch(combined_input_batch)


def test_converter__serialize_dataset(
    converter: PgmJsonConverter, pgm_input_data: SingleDataset, pgm_batch_data: BatchDataset
):
    with pytest.raises(ValueError, match="Invalid data format"):
        converter._serialize_dataset(data={"node": "attribute"}, extra_info=None)  # type: ignore
    with pytest.raises(ValueError, match="Invalid data format"):
        converter._serialize_dataset(data=pgm_batch_data, extra_info=None)

    structured_data = converter._serialize_dataset(data=pgm_input_data, extra_info=None)
    assert structured_data == {"node": [{"id": 1}, {"id": 2}]}

    extra_info = {1: {"dummy": "data"}}
    structured_data_with_extra_info = converter._serialize_dataset(data=pgm_input_data, extra_info=extra_info)
    assert structured_data_with_extra_info == {"node": [{"id": 1, "dummy": "data"}, {"id": 2}]}
