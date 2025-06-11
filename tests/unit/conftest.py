# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from contextlib import suppress

import pandas as pd

with suppress(pd.errors.OptionError):
    # TODO(mgovers) We're ready for Pandas 3.x, but pandapower is not. Move to parent conftest when it is.
    pd.set_option("mode.copy_on_write", True)
