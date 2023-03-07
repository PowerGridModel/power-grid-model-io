# SPDX-FileCopyrightText: 2022 Contributors to the Power Grid Model project <dynamic.grid.calculation@alliander.com>
#
# SPDX-License-Identifier: MPL-2.0
"""
Helper function to extract zip files

csv_dir_path = extract("/tmp/1-complete_data-mixed-all-0-sw.zip")

This extracts the files, in a folder which corresponds to the zip file name ("/tmp/1-complete_data-mixed-all-0-sw/" in
our example), and it returns the path to that directory. By default, it will not re-download or re-extract the zip
file as long as the files exist.

"""

import zipfile
from pathlib import Path
from typing import Optional

import structlog
from tqdm import tqdm

_log = structlog.get_logger(__name__)


def extract(src_file_path: Path, dst_dir_path: Optional[Path] = None, skip_if_exists=False) -> Path:
    """
    Extract a .zip file and return the destination dir

    Args:
        src_file_path: The .zip file to extract.
        dst_dir_path: An optional destination path. If none is given, the src_file_path without .zip extension is used.
        skip_if_exists: Skip existing files, otherwise raise an exception when a file exists.

    Returns: The path where the files are extracted

    """
    if src_file_path.suffix.lower() != ".zip":
        raise ValueError(f"Only files with .zip extension are supported, got {src_file_path.name}")

    if dst_dir_path is None:
        dst_dir_path = src_file_path.with_suffix("")

    log = _log.bind(src_file_path=src_file_path, dst_dir_path=dst_dir_path)

    if dst_dir_path.exists():
        if not dst_dir_path.is_dir():
            raise NotADirectoryError(f"Destination dir {dst_dir_path} exists and is not a directory")

    # Create the destination directory
    dst_dir_path.mkdir(parents=True, exist_ok=True)

    # Extract per file, so we can show a progress bar
    with zipfile.ZipFile(src_file_path, "r") as zip_file:
        file_list = zip_file.namelist()
        for file_path in tqdm(desc="Extracting", iterable=file_list, total=len(file_list), unit="file", leave=True):
            dst_file_path = dst_dir_path / file_path
            if dst_file_path.exists() and dst_file_path.stat().st_size > 0:
                if skip_if_exists:
                    log.debug("Skip file extraction, destination file exists", dst_file_path=dst_file_path)
                    continue
                raise FileExistsError(f"Destination file {dst_dir_path / file_path} exists and is not empty")
            zip_file.extract(member=file_path, path=dst_dir_path)

    # Zip files often contain a single directory with the same name as the zip file.
    # In that case, return the dir to that directory instead of the root dir
    only_item: Optional[Path] = None
    for item in dst_dir_path.iterdir():
        # If only_item is None, this is the first iteration, so item may be the only item
        if only_item is None:
            only_item = item
        # Else, if only_item is not None, there are more than one items in the root of the directory.
        # This means that there is no 'only_item' and we can stop the loop
        else:
            only_item = None
            break
    if only_item and only_item.is_dir() and only_item.name == src_file_path.stem:
        dst_dir_path = only_item

    return dst_dir_path.resolve()
