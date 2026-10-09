# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0

from contextlib import suppress
from importlib.metadata import version
from importlib.util import find_spec

import pandas as pd
from packaging.version import Version

if find_spec("pandapower") is None:
    # these modules import pandapower at module level
    collect_ignore = [
        "validation/converters/test_pandapower_converter_input.py",
        "validation/converters/test_pandapower_converter_output.py",
    ]

if Version(version("pandas")) < Version("3.0.0"):
    # Opt-in to Pandas 3 behavior for Pandas 2.x
    with suppress(pd.errors.OptionError):
        pd.options.future.no_silent_downcasting = True
