# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from contextlib import suppress
from importlib.metadata import version

import pandas as pd
from packaging.version import Version

if Version(version("pandas")) < Version("3.0.0"):
    # Opt-in to Pandas 3 behavior for Pandas 2.x
    no_silent_downcasting_option = True
    with suppress(pd.errors.OptionError):
        pd.set_option("future.no_silent_downcasting", no_silent_downcasting_option)

    copy_on_write_option = False
    with suppress(pd.errors.OptionError):
        pd.set_option("mode.copy_on_write", copy_on_write_option)
