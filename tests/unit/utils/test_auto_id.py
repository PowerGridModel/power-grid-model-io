# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
from pytest import raises

from power_grid_model_io.utils.auto_id import AutoID


def test_auto_id__without_items():
    auto_id = AutoID()
    assert auto_id() == 0
    assert auto_id() == 1
    assert auto_id() == 2
    assert auto_id[0] == 0
    assert auto_id[1] == 1
    assert auto_id[2] == 2
    with raises(IndexError):
        _ = auto_id[3]

    assert 0 in auto_id
    assert 1 in auto_id
    assert 2 in auto_id
    assert 3 not in auto_id


def test_auto_id__with_hashable_items():
    auto_id = AutoID()
    assert auto_id(item="Alpha") == 0
    assert auto_id(item="Bravo") == 1
    assert auto_id(item="Alpha") == 0  # because key "Alpha" already existed
    assert auto_id[0] == "Alpha"
    assert auto_id[1] == "Bravo"
    with raises(IndexError):
        _ = auto_id[2]

    assert 0 in auto_id
    assert 1 in auto_id
    assert 2 not in auto_id
    assert "Alpha" in auto_id
    assert "Bravo" in auto_id
    assert "Charly" not in auto_id


def test_auto_id__with_non_hashable_items():
    auto_id = AutoID()
    with raises(TypeError, match="Unhashable type: 'dict'"):
        auto_id(item={"name": "Alpha"})
    assert auto_id(item={"name": "Alpha"}, key="Alpha") == 0
    assert auto_id(item={"name": "Bravo"}, key="Bravo") == 1
    assert auto_id(item={"name": "Alpha"}, key="Alpha") == 0  # because key "Alpha" already existed
    assert auto_id[0] == {"name": "Alpha"}
    assert auto_id[1] == {"name": "Bravo"}
    with raises(IndexError):
        _ = auto_id[2]

    assert 0 in auto_id
    assert 1 in auto_id
    assert 2 not in auto_id
    assert "Alpha" in auto_id
    assert "Bravo" in auto_id
    assert "Charly" not in auto_id
    assert {"name": "Alpha"} not in auto_id
    assert {"name": "Bravo"} not in auto_id


def test_auto_id__with_clashing_keys():
    auto_id = AutoID()
    assert auto_id(item={"name": "Alpha"}, key="Alpha") == 0
    assert auto_id(item={"name": "Bravo"}, key="Bravo") == 1
    assert auto_id(item={"name": "Charly"}, key="Alpha") == 0  # because key "Alpha" already existed
    assert auto_id[0] == {"name": "Charly"}  # Note that the item was overwritten silently
    assert auto_id[1] == {"name": "Bravo"}
    with raises(IndexError):
        _ = auto_id[2]
