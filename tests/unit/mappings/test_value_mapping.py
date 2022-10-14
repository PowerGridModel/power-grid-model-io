# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock, patch

import pytest
from pytest import fixture

from power_grid_model_io.mappings.value_mapping import ValueMapping, Values


@fixture
def mapping() -> ValueMapping:
    return ValueMapping({".*\.switch_state": {"off": 0, "in": 1}, "N1": {"none": False, "own": True}})


def test_get_substitutions__exact_match(mapping: ValueMapping):
    assert mapping.get_substitutions("N1") == {"none": False, "own": True}


def test_get_substitutions__regex_match(mapping: ValueMapping):
    assert mapping.get_substitutions("from_node.switch_state") == {"off": 0, "in": 1}


def test_get_substitutions__no_match(mapping: ValueMapping):
    with pytest.raises(KeyError):
        mapping.get_substitutions("N2")
