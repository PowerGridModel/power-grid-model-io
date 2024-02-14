# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd

try:
    pd.set_option("future.no_silent_downcasting", True)
except pd.errors.OptionError:
    pass
