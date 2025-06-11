# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from contextlib import suppress

import pandas as pd

with suppress(pd.errors.OptionError):
    pd.set_option("future.no_silent_downcasting", True)
