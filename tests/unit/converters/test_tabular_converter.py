# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model IO project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

from typing import Optional, Tuple

import pytest

from power_grid_model_io.converters.tabular_converter import COL_REF_RE, OPT_COL_RE


def ref_cases():
    yield "OtherSheet!ValueColumn[IdColumn=RefColumn]", (
        "OtherSheet",
        "ValueColumn",
        None,
        None,
        "IdColumn",
        None,
        None,
        "RefColumn",
    )

    yield "OtherSheet!ValueColumn[OtherSheet!IdColumn=ThisSheet!RefColumn]", (
        "OtherSheet",
        "ValueColumn",
        "OtherSheet!",
        "OtherSheet",
        "IdColumn",
        "ThisSheet!",
        "ThisSheet",
        "RefColumn",
    )

    yield "OtherSheet.ValueColumn[IdColumn=RefColumn]", None
    yield "ValueColumn[IdColumn=RefColumn]", None
    yield "OtherSheet![IdColumn=RefColumn]", None


@pytest.mark.parametrize("value,groups", ref_cases())
def test_col_ref_pattern(value: str, groups: Optional[Tuple[Optional[str]]]):
    match = COL_REF_RE.fullmatch(value)
    if groups is None:
        assert match is None
    else:
        assert match is not None
        assert match.groups() == groups
