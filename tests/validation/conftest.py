# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from contextlib import suppress
from importlib.metadata import version

import pandas as pd
from packaging.version import Version

if Version(version("pandas")) < Version("3.0.0"):
    # Opt-in to Pandas 3 behavior for Pandas 2.x
    with suppress(pd.errors.OptionError):
        pd.options.future.no_silent_downcasting = True

    with suppress(pd.errors.OptionError):
        pd.options.mode.copy_on_write = False
