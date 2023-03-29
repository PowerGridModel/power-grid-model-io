# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0

import os
import re
from pathlib import Path

# noinspection PyPackageRequirements
import requests
from setuptools import setup


def set_long_description(pkg_dir: Path):
    with open(pkg_dir / "README.md", "r") as f:
        raw_readme = f.read()
    if "GITHUB_SHA" not in os.environ:
        readme = raw_readme
    else:
        sha = os.environ["GITHUB_SHA"].lower()
        url = f"https://github.com/PowerGridModel/power-grid-model-io/blob/{sha}/"
        readme = re.sub(r"(\[[^\(\)\[\]]+\]\()((?!http)[^\(\)\[\]]+\))", f"\\1{url}\\2", raw_readme)
    with open("README.md", "w") as f:
        f.write(readme)


def set_version(pkg_dir: Path):
    with open(pkg_dir / "VERSION") as f:
        version = f.read().strip().strip("\n")
    major, minor = (int(x) for x in version.split("."))
    latest_major, latest_minor, latest_patch = get_pypi_latest()
    # get version
    version = get_new_version(major, minor, latest_major, latest_minor, latest_patch)
    # mutate version in GitHub Actions
    if ("GITHUB_SHA" in os.environ) and ("GITHUB_REF" in os.environ) and ("GITHUB_RUN_NUMBER" in os.environ):
        sha = os.environ["GITHUB_SHA"]
        ref = os.environ["GITHUB_REF"]
        build_number = os.environ["GITHUB_RUN_NUMBER"]
        # short hash number in numeric
        short_hash = f'{int(f"0x{sha[0:6]}", base=16):08}'

        if "main" in ref:
            # main branch
            # major.minor.patch
            # do nothing
            pass
        elif "release" in ref:
            # release branch
            # major.minor.patch rc 9 build_number short_hash
            # NOTE: the major.minor in release branch is usually higher than the main branch
            # this is the leading version if you enable test version in pip install
            version += f"rc9{build_number}{short_hash}"
        else:
            # feature branch
            # major.minor.patch a 1 build_number short_hash
            version += f"a1{build_number}{short_hash}"
    with open(pkg_dir / "PYPI_VERSION", "w") as f:
        f.write(version)


def get_pypi_latest():
    r = requests.get("https://pypi.org/pypi/power-grid-model-io/json")
    if r.status_code == 404:
        return 0, 0, 0
    data = r.json()
    version = str(data["info"]["version"])
    return (int(x) for x in version.split("."))


def get_new_version(major, minor, latest_major, latest_minor, latest_patch):
    if (major > latest_major) or ((major == latest_major) and minor > latest_minor):
        # brand-new version with patch zero
        return f"{major}.{minor}.0"
    elif major == latest_major and minor == latest_minor:
        # current version, increment path
        return f"{major}.{minor}.{latest_patch + 1}"
    else:
        # does not allow building older version
        raise ValueError(
            "Invalid version number!\n"
            f"latest version: {latest_major}.{latest_minor}.{latest_patch}\n"
            f"to be built version: {major}.{minor}\n"
        )


def prepare_pkg(setup_file: Path):
    """

    Args:
        setup_file:
    Returns:

    """
    pkg_dir = setup_file.parent
    set_version(pkg_dir)
    set_long_description(pkg_dir)


prepare_pkg(Path(__file__).resolve())
setup()
