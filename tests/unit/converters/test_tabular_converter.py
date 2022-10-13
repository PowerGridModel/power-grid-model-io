# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pathlib import Path

import pandas as pd
from power_grid_model import initialize_array
from power_grid_model.data_types import SingleDataset
from typing import Optional, Tuple
from unittest.mock import MagicMock

import pytest

from power_grid_model_io.converters.tabular_converter import COL_REF_RE, TabularConverter
from power_grid_model_io.data_types import TabularData

MAPPING_FILE = Path(__file__).parent / "test_data/mapping.yaml"


def ref_cases():
    yield "OtherTable!ValueColumn[IdColumn=RefColumn]", (
        "OtherTable",
        "ValueColumn",
        None,
        None,
        "IdColumn",
        None,
        None,
        "RefColumn",
    )

    yield "OtherTable!ValueColumn[OtherTable!IdColumn=ThisTable!RefColumn]", (
        "OtherTable",
        "ValueColumn",
        "OtherTable!",
        "OtherTable",
        "IdColumn",
        "ThisTable!",
        "ThisTable",
        "RefColumn",
    )

    yield "OtherTable.ValueColumn[IdColumn=RefColumn]", None
    yield "ValueColumn[IdColumn=RefColumn]", None
    yield "OtherTable![IdColumn=RefColumn]", None


@pytest.mark.parametrize("value,groups", ref_cases())
def test_col_ref_pattern(value: str, groups: Optional[Tuple[Optional[str]]]):
    match = COL_REF_RE.fullmatch(value)
    if groups is None:
        assert match is None
    else:
        assert match is not None
        assert match.groups() == groups


@pytest.fixture
def converter():
    converter = TabularConverter(mapping_file=MAPPING_FILE)
    return converter


@pytest.fixture
def pgm_input_data():
    node = initialize_array("input", "node", 2)
    return {"node": node}


@pytest.fixture
def tabular_data():
    nodes = pd.DataFrame(data=[[1, 10.5], [2, 0.4]], columns=pd.MultiIndex.from_tuples([("id_number", ""), ("u_nom", "kV")]))
    lines = pd.DataFrame(data=[[1, 100], [2, 200]], columns=["id_number", "from_node_side"])
    tabular_data = TabularData(nodes=nodes, lines=lines)
    return tabular_data


def test_converter__set_mapping_file(converter: TabularConverter):
    with pytest.raises(ValueError, match="Mapping file should be a .yaml file, .txt provided."):
        converter.set_mapping_file(mapping_file=Path("dummy/path.txt"))

    dummy_path = Path(__file__).parent / "test_data/dummy_mapping.yaml"
    with pytest.raises(KeyError, match="Missing 'grid' mapping in mapping_file"):
        converter.set_mapping_file(mapping_file=dummy_path)

    converter.set_mapping_file(mapping_file=MAPPING_FILE)


def test_converter__parse_data(converter: TabularConverter, tabular_data: TabularData):
    data = MagicMock()
    converter._parse_data(data=data, data_type="dummy")
    data.set_unit_multipliers.assert_called_once()
    data.set_substitutions.assert_called_once()

    pgm_input_data = converter._parse_data(data=tabular_data, data_type="input")
    assert list(pgm_input_data.keys()) == ["node", "line"]
    assert len(pgm_input_data["node"]) == 2
    assert (pgm_input_data["node"]["id"] == [0, 1]).all()
    assert (pgm_input_data["node"]["u_rated"] == [10.5e3, 400]).all()

    assert len(pgm_input_data["line"]) == 2
    assert (pgm_input_data["line"]["id"] == [2, 3]).all()
    # TODO: fix line below
    # assert (pgm_input_data["line"]["from_node"] == [0, 1]).all()


def test_converter__convert_table_to_component():
    # TODO
    pass


def test_converter__convert_col_def_to_attribute():
    # TODO
    pass


def test_converter__handle_column():
    # TODO
    pass


def test_converter__handle_id_column():
    # TODO
    pass


def test_converter__handle_extra_info():
    # TODO
    pass


def test_converter__handle_node_ref_column():
    # TODO
    pass


def test_converter__merge_pgm_data():
    # TODO
    pass


def test_converter__serialize_data(converter: TabularConverter, pgm_input_data: SingleDataset):
    with pytest.raises(NotImplementedError, match=r"Extra info can not \(yet\) be stored for tabular data"):
        converter._serialize_data(data=pgm_input_data, extra_info={})
    with pytest.raises(NotImplementedError, match=r"Batch data can not \(yet\) be stored for tabular data"):
        converter._serialize_data(data=[])  # type: ignore
    # TODO: serialize_data expects pgm Dataset, TabularData, expects pd.DataFrame
    # tabular_data = converter._serialize_data(data=pgm_input_data)

def test_converter__id_lookup():
    # TODO
    pass


def test_converter__parse_col_def():
    # TODO
    pass


def test_converter__parse_col_def_const():
    # TODO
    pass


def test_converter__parse_col_def_column_name():
    # TODO
    pass


def test_converter__parse_col_def_column_reference():
    # TODO
    pass


def test_converter__parse_col_def_function():
    # TODO
    pass


def test_converter__parse_col_def_composite():
    # TODO
    pass
