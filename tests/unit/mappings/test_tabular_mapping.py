# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from pytest import fixture, raises

from power_grid_model_io.mappings.tabular_mapping import TabularMapping


@fixture
def mapping() -> TabularMapping:
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
    mapping = TabularMapping({"Nodes": [{"node": {"id": "ID"}}]})  # type: ignore

    # Act and Assert
    with raises(TypeError):
        next(mapping.instances(table="Nodes"))
