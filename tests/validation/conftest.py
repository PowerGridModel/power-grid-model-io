# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

import pandas as pd

try:
    pd.set_option("future.no_silent_downcasting", True)
except pd.errors.OptionError:
    pass

try:
    # TODO(mgovers) We're ready for Pandas 3.x, but pandapower is not. Move to parent conftest when it is.
    pd.set_option("mode.copy_on_write", False)
except pd.errors.OptionError:
    pass
