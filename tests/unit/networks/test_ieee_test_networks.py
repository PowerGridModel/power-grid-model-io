# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from unittest.mock import MagicMock, patch

import pytest

from power_grid_model_io.networks import IeeeFourBusVariant, ieee_four_bus


@pytest.fixture
def mock_json_deserialize():
    """Mock the json_deserialize_from_file utility."""
    with patch("power_grid_model_io.networks.ieee_test_networks.json_deserialize_from_file") as mock_func:
        mock_func.return_value = MagicMock()
        yield mock_func


def test_ieee_four_bus_default(mock_json_deserialize):
    """Test calling ieee_four_bus with default parameters."""

    # Act
    result = ieee_four_bus()

    # Assert
    assert result == mock_json_deserialize.return_value
    mock_json_deserialize.assert_called_once()

    # Verify input path points to the default scenario input.json file
    call_path = mock_json_deserialize.call_args[0][0]
    assert call_path.name == "input.json"
    assert str(IeeeFourBusVariant.step_down_dyn_balanced_loading) in str(call_path)


@pytest.mark.parametrize("variant", list(IeeeFourBusVariant) + [s.value for s in IeeeFourBusVariant])
def test_ieee_four_bus_valid_enum(mock_json_deserialize, variant):
    """Test calling ieee_four_bus with a valid IeeeFourBusScenario enum member."""

    # Act
    result = ieee_four_bus(variant=variant)

    # Assert
    assert result == mock_json_deserialize.return_value
    mock_json_deserialize.assert_called_once()

    call_path = mock_json_deserialize.call_args[0][0]
    assert call_path.name == "input.json"
    assert str(variant) in str(call_path)


@pytest.mark.parametrize(
    "invalid_variant",
    [
        "invalid_variant_name",
        "",
        12345,
        None,
    ],
)
def test_ieee_four_bus_invalid_scenario_raises_value_error(mock_json_deserialize, invalid_variant):
    """Test that invalid scenarios raise a ValueError and skip deserialization."""
    # Act
    with pytest.raises(ValueError, match=f"Invalid variant: {invalid_variant!r}"):
        ieee_four_bus(variant=invalid_variant)

    # Assert
    mock_json_deserialize.assert_not_called()
