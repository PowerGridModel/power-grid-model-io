# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from enum import StrEnum


class IeeeFourBusScenario(StrEnum):
    # network scenarios
    step_down_dyn_balanced_load = "step_down_dyn_balanced_load"
    step_down_dyn_unbalanced_load = "step_down_dyn_unbalanced_load"
    step_down_ynyn_balanced_load = "step_down_ynyn_balanced_load"
    step_down_ynyn_unbalanced_load = "step_down_ynyn_unbalanced_load"
    step_down_ynd_balanced_load = "step_down_ynd_balanced_load"
    step_down_ynd_unbalanced_load = "step_down_ynd_unbalanced_load"
    step_down_yd_balanced_load = "step_down_yd_balanced_load"
    step_down_yd_unbalanced_load = "step_down_yd_unbalanced_load"
    step_down_dd_balanced_load = "step_down_dd_balanced_load"
    step_down_dd_unbalanced_load = "step_down_dd_unbalanced_load"
    step_up_dyn_balanced_load = "step_up_dyn_balanced_load"
    step_up_dyn_unbalanced_load = "step_up_dyn_unbalanced_load"
    step_up_ynyn_balanced_load = "step_up_ynyn_balanced_load"
    step_up_ynyn_unbalanced_load = "step_up_ynyn_unbalanced_load"
    step_up_ynd_balanced_load = "step_up_ynd_balanced_load"
    step_up_ynd_unbalanced_load = "step_up_ynd_unbalanced_load"
    step_up_yd_balanced_load = "step_up_yd_balanced_load"
    step_up_yd_unbalanced_load = "step_up_yd_unbalanced_load"
    step_up_dd_balanced_load = "step_up_dd_balanced_load"
    step_up_dd_unbalanced_load = "step_up_dd_unbalanced_load"
