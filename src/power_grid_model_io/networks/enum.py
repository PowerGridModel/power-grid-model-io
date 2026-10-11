# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from enum import StrEnum


class IeeeFourBusVariant(StrEnum):
    """
    variants for the IEEE 4-bus standard test feeder.

    Combines transformer type (step-up/down), vector groups, and loadinging (balanced/unbalanced).
    """

    step_down_dyn_balanced_loading = "step_down_dyn_balanced_loading"
    step_down_dyn_unbalanced_loading = "step_down_dyn_unbalanced_loading"
    step_down_ynyn_balanced_loading = "step_down_ynyn_balanced_loading"
    step_down_ynyn_unbalanced_loading = "step_down_ynyn_unbalanced_loading"
    step_down_ynd_balanced_loading = "step_down_ynd_balanced_loading"
    step_down_ynd_unbalanced_loading = "step_down_ynd_unbalanced_loading"
    step_down_yd_balanced_loading = "step_down_yd_balanced_loading"
    step_down_yd_unbalanced_loading = "step_down_yd_unbalanced_loading"
    step_down_dd_balanced_loading = "step_down_dd_balanced_loading"
    step_down_dd_unbalanced_loading = "step_down_dd_unbalanced_loading"
    step_up_dyn_balanced_loading = "step_up_dyn_balanced_loading"
    step_up_dyn_unbalanced_loading = "step_up_dyn_unbalanced_loading"
    step_up_ynyn_balanced_loading = "step_up_ynyn_balanced_loading"
    step_up_ynyn_unbalanced_loading = "step_up_ynyn_unbalanced_loading"
    step_up_ynd_balanced_loading = "step_up_ynd_balanced_loading"
    step_up_ynd_unbalanced_loading = "step_up_ynd_unbalanced_loading"
    step_up_yd_balanced_loading = "step_up_yd_balanced_loading"
    step_up_yd_unbalanced_loading = "step_up_yd_unbalanced_loading"
    step_up_dd_balanced_loading = "step_up_dd_balanced_loading"
    step_up_dd_unbalanced_loading = "step_up_dd_unbalanced_loading"
