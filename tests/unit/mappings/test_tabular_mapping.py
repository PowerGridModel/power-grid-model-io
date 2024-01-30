# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
import pytest
from pytest import fixture

from power_grid_model_io.mappings.tabular_mapping import TabularMapping


@fixture
def mapping() -> TabularMapping:
    # Arrange
    return TabularMapping(
        {
            "Nodes": {"node": {"id": "ID"}},
            "Cables": {
                "line": {
                    "id": "ID",
                    "from_node": "FROM_NODE_ID",
                    "to_node": "TO_NODE_ID",
                }
            },
            "Generator": {
                "generator": {
                    "id": "ID",
                    "from_node": "FROM_NODE_ID",
                    "to_node": "TO_NODE_ID",
                },
                "node": [
                    {
                        "id": "FROM_NODE_ID",
                    },
                    {
                        "id": "TO_NODE_ID",
                    },
                ],
            },
        }
    )


def test_tables(mapping: TabularMapping):
    # Act
    actual = list(mapping.tables())

    # Assert
    expected = ["Nodes", "Cables", "Generator"]
    assert actual == expected


def test_instances(mapping: TabularMapping):
    # Act
    actual = list(mapping.instances(table="Generator"))

    # Assert
    expected = [
        ("generator", {"id": "ID", "from_node": "FROM_NODE_ID", "to_node": "TO_NODE_ID"}),
        ("node", {"id": "FROM_NODE_ID"}),
        ("node", {"id": "TO_NODE_ID"}),
    ]
    assert actual == expected


def test_instances__exception():
    # Arrange
    mapping = TabularMapping({"Nodes": [1, 2, 3]})  # type: ignore

    # Act / Assert
    with pytest.raises(TypeError, match="Invalid table mapping for Nodes; expected a dictionary got list"):
        next(mapping.instances(table="Nodes"))
