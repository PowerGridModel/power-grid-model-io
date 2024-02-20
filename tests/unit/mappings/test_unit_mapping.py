# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
import logging
from unittest.mock import MagicMock, patch

import pytest
import structlog
from pytest import fixture

from power_grid_model_io.mappings.unit_mapping import UnitMapping, Units


@fixture
def units() -> Units:
    return {"A": None, "W": {"kW": 1000.0, "MW": 1000000.0}}


@patch("power_grid_model_io.mappings.unit_mapping.UnitMapping.set_mapping")
def test_constructor(mock_set_mapping: MagicMock, units: Units):
    # Act
    UnitMapping(units)

    # Assert
    mock_set_mapping.assert_called_once_with(mapping=units)


def test_set_mapping(units: Units):
    # Arrange
    mapping = UnitMapping()

    # Act / Assert (no exceptions)
    mapping.set_mapping(mapping=units)


def test_set_mapping__multiple_mappings():
    # Arrange
    mapping = UnitMapping()
    units = {"X": {"A": 111.0}, "Y": {"A": 222.0}}

    # Act / Assert
    with pytest.raises(ValueError, match="Multiple unit definitions for 'A': 1A = 111.0X = 222.0Y"):
        mapping.set_mapping(mapping=units)


def test_set_mapping__valid_si_to_si_mapping():
    # Arrange
    mapping = UnitMapping()
    units = {"A": {"A": 1.0}}

    # Act / Assert (no exceptions)
    mapping.set_mapping(mapping=units)


def test_set_mapping__invalid_si_to_si_mapping():
    # Arrange
    mapping = UnitMapping()
    units = {"A": {"A": 2.0}}

    # Act / Assert
    with pytest.raises(ValueError, match="Invalid unit definition for 'A': 1A cannot be 2.0A"):
        mapping.set_mapping(mapping=units)


def test_get_unit_multiplier(units: Units):
    # Arrange
    mapping = UnitMapping(mapping=units)

    # Act / Assert
    assert mapping.get_unit_multiplier("A") == (1.0, "A")
    assert mapping.get_unit_multiplier("W") == (1.0, "W")
    assert mapping.get_unit_multiplier("kW") == (1000.0, "W")
    assert mapping.get_unit_multiplier("MW") == (1000000.0, "W")

    with pytest.raises(KeyError):
        mapping.get_unit_multiplier("I")


def test_mapping_logger():
    log_level = logging.DEBUG
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    mapping = UnitMapping(
        logger=structlog.wrap_logger(logger, wrapper_class=structlog.make_filtering_bound_logger(log_level))
    )
    assert mapping._log._logger.getEffectiveLevel() == log_level
