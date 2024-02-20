# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
import logging

import pytest
import structlog

from power_grid_model_io.mappings.field_mapping import FieldMapping


def test_mapping_logger():
    log_level = logging.DEBUG
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level)
    mapping = FieldMapping(
        logger=structlog.wrap_logger(logger, wrapper_class=structlog.make_filtering_bound_logger(log_level))
    )
    assert mapping._log._logger.getEffectiveLevel() == log_level
