# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from enum import StrEnum


class IEEE_four_bus_scenario(StrEnum):
    # network scenarios
    step_down_dyn1_balanced_load = "step_down_dyn1_balanced_load"
    step_down_dyn1_unbalanced_load = "step_down_dyn1_unbalanced_load"
    step_down_ynyn0_balanced_load = "step_down_ynyn0_balanced_load"
    step_down_ynyn0_unbalanced_load = "step_down_ynyn0_unbalanced_load"
    step_down_yd1_balanced_load = "step_down_yd1_balanced_load"
    step_down_yd1_unbalanced_load = "step_down_yd1_unbalanced_load"
    step_down_dd0_balanced_load = "step_down_dd0_balanced_load"
    step_down_dd0_unbalanced_load = "step_down_dd0_unbalanced_load"
    step_up_dyn1_balanced_load = "step_up_dyn1_balanced_load"
    step_up_dyn1_unbalanced_load = "step_up_dyn1_unbalanced_load"
    step_up_ynyn0_balanced_load = "step_up_ynyn0_balanced_load"
    step_up_ynyn0_unbalanced_load = "step_up_ynyn0_unbalanced_load"
