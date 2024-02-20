# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
import logging

import pytest
import structlog
from pytest import fixture

from power_grid_model_io.mappings.value_mapping import ValueMapping


@fixture
def mapping() -> ValueMapping:
    return ValueMapping(
        {
            "sources.closed": {"yes": 1, "no": 0},
            ".*_switch_state": {"off": 0, "in": 1, "on": 1},
            "N1": {"none": False, "own": True},
        }
    )


def test_get_substitutions__exact_match(mapping: ValueMapping):
    assert mapping.get_substitutions("N1") == {"none": False, "own": True}


def test_get_substitutions__exact_match_with_table(mapping: ValueMapping):
    assert mapping.get_substitutions("closed", table="sources") == {"yes": 1, "no": 0}
    with pytest.raises(KeyError):
        mapping.get_substitutions("closed")


def test_get_substitutions__regex_match(mapping: ValueMapping):
    assert mapping.get_substitutions("from_switch_state") == {"off": 0, "in": 1, "on": 1}


def test_get_substitutions__no_match(mapping: ValueMapping):
    with pytest.raises(KeyError):
        mapping.get_substitutions("N2")


def test_mapping_logger():
    log_level = logging.DEBUG
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    mapping = ValueMapping(
        logger=structlog.wrap_logger(logger, wrapper_class=structlog.make_filtering_bound_logger(log_level))
    )
    assert mapping._log._logger.getEffectiveLevel() == log_level
