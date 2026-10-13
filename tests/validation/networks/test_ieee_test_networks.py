# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import pytest
from power_grid_model import CalculationType
from power_grid_model.validation import assert_valid_input_data

from power_grid_model_io.networks import IeeeFourNodeVariant, ieee_four_node


@pytest.mark.parametrize("variant", list(IeeeFourNodeVariant))
def test_ieee_four_node_data_validation(variant):
    """Load input data by calling ieee_four_node and validate the input data."""

    # Act
    input_data = ieee_four_node(variant=variant)

    # Assert
    assert_valid_input_data(input_data=input_data, calculation_type=CalculationType.power_flow, symmetric=False)
