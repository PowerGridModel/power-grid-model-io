# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from pytest import fixture

from power_grid_model_io.mappings.multiplier_mapping import MultiplierMapping


@fixture
def mapping() -> MultiplierMapping:
    return MultiplierMapping(
        {
            "sources.p": 1.0,
            ".*_power": 2.0,
            "Q": 3.0,
        }
    )


def test_get_multiplier__exact_match(mapping: MultiplierMapping):
    assert mapping.get_multiplier("Q") == 3.0


def test_get_multiplier__exact_match_with_table(mapping: MultiplierMapping):
    assert mapping.get_multiplier("p", table="sources") == 1.0
    with pytest.raises(KeyError):
        mapping.get_multiplier("p")


def test_get_multiplier__regex_match(mapping: MultiplierMapping):
    assert mapping.get_multiplier("measured_power") == 2.0


def test_get_multiplier__no_match(mapping: MultiplierMapping):
    with pytest.raises(KeyError):
        mapping.get_multiplier("X")
