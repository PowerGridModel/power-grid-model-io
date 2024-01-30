# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
"""
Helper functions to download (and store) files from the internet

The most simple (and intended) usage is:
url = "http://141.51.193.167/simbench/gui/usecase/download/?simbench_code=1-complete_data-mixed-all-0-sw&format=csv"
zip_file_path = download(url)

It will download the zip file 1-complete_data-mixed-all-0-sw.zip to a folder in you systems temp dir; for example
"/tmp/1-complete_data-mixed-all-0-sw.zip".

Another convenience function is download_and_extract():

csv_dir_path = download_and_extract(url)

This downloads the zip file as described above, and then it extracts the files there as well, in a folder which
corresponds to the zip file name ("/tmp/1-complete_data-mixed-all-0-sw/" in our example), and it returns the path to
that directory. By default, it will not re-download or re-extract the zip file as long as the files exist in your
temp dir. Your temp dir is typically emptied when you reboot your computer.

"""

import base64
import hashlib
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from shutil import rmtree as remove_dir
from typing import Optional, Union
from urllib import request

import structlog
from tqdm import tqdm

from power_grid_model_io.utils.zip import extract

_log = structlog.get_logger(__name__)


@dataclass
class ResponseInfo:
    """
    Data class to store response information extracted from the response header
    """

    status: int
    file_name: Optional[str] = None
    file_size: Optional[int] = None


class DownloadProgressHook:  # pylint: disable=too-few-public-methods
    """
    Report hook for request.urlretrieve() to update a progress bar based on the amount of downloaded blocks
    """

    def __init__(self, progress_bar: tqdm):
        """
        Report hook for request.urlretrieve() to update a progress bar based on the amount of downloaded blocks

        Args:
            progress_bar: A tqdm progress bar
        """
        self._progress_bar = progress_bar
        self._last_block = 0

    def __call__(self, block_num: int, block_size: int, file_size: int) -> None:
        """
        Args:
            block_num: The last downloaded block number
            block_size: The block size in bytes
            file_size: The file size in bytes (may be 0 in the first call)

        """
        if file_size > 0:
            self._progress_bar.total = file_size
        self._progress_bar.update((block_num - self._last_block) * block_size)
        self._last_block = block_num


def download_and_extract(
    url: str, dir_path: Optional[Path] = None, file_name: Optional[Union[str, Path]] = None, overwrite: bool = False
) -> Path:
    """
    Download a file from a URL and store it locally, extract the contents and return the path to the contents.

    Args:
        url:       The url to the .zip file
        dir_path:  An optional dir path to store the downloaded file. If no dir_path is given the current working dir
                   will be used.
        file_name: An optional file name (or path relative to dir_path). If no file_name is given, a file name is
                   generated based on the url
        overwrite: Should we download the file, even if we have downloaded already (and the file size still matches)?
                   Be careful with this option, as it will remove files from your drive irreversibly!

    Returns:
        The path to the downloaded file
    """

    # Download the file and use the file name as the base name for the extraction directory
    src_file_path = download(url=url, file_name=file_name, dir_path=dir_path, overwrite=overwrite)
    dst_dir_path = src_file_path.with_suffix("")

    # If we explicitly want to overwrite the extracted files, remove the destination dir.
    if overwrite and dst_dir_path.is_dir():
        remove_dir(dst_dir_path)

    # Extract the files and return the path of the extraction directory
    return extract(src_file_path=src_file_path, dst_dir_path=dst_dir_path, skip_if_exists=not overwrite)


def download(
    url: str, file_name: Optional[Union[str, Path]] = None, dir_path: Optional[Path] = None, overwrite: bool = False
) -> Path:
    """
    Download a file from a URL and store it locally

    Args:
        url:       The url to the file
        file_name: An optional file name (or path relative to dir_path). If no file_name is given, a file name is
                   generated based on the url
        dir_path:  An optional dir path to store the downloaded file. If no dir_path is given the current working dir
                   will be used.
        overwrite: Should we download the file, even if we have downloaded already (and the file size still matches)?

    Returns:
        The path to the downloaded file
    """

    # get the response info, if the status is not 200
    info = get_response_info(url=url)
    if info.status != 200:
        raise IOError(f"Could not download from URL, status={info.status}")

    if file_name is None and info.file_name:
        file_name = info.file_name

    file_path = get_download_path(dir_path=dir_path, file_name=file_name, unique_key=url)
    log = _log.bind(url=url, file_path=file_path)

    if file_path.is_file():
        if overwrite:
            log.debug("Forced re-downloading existing file")
            # Don't remove the existing file just yet... Let's first see if we can download a new version.
        else:
            local_size = file_path.stat().st_size
            if local_size == info.file_size:
                log.debug("Skip downloading existing file")
                return file_path
            log.debug(
                "Re-downloading existing file, because the size has changed",
                local_size=local_size,
                remote_size=info.file_size,
            )
    else:
        log.debug("Downloading file")

    # Download to a temp file first, so the results are not stored if the transfer fails
    with tqdm(desc="Downloading", unit="B", unit_scale=True, leave=True) as progress_bar:
        report_hook = DownloadProgressHook(progress_bar)
        temp_file, _headers = request.urlretrieve(url, reporthook=report_hook)

    # Check if the file contains any content
    temp_path = Path(temp_file)
    if temp_path.stat().st_size == 0:
        log.warning("Downloaded an empty file")

    # Remove the file, if it already exists
    file_path.unlink(missing_ok=True)

    # Move the file to it's final destination
    file_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path.rename(file_path)
    log.debug("Downloaded file", file_size=file_path.stat().st_size)

    return file_path


def get_response_info(url: str) -> ResponseInfo:
    """
    Retrieve the file size of a given URL (based on it's header)

    Args:
        url: The url to the file

    Return:
        The file size in bytes
    """
    with request.urlopen(url) as context:
        status = context.status
        headers = context.headers
    file_size = int(headers["Content-Length"]) if "Content-Length" in headers else None
    matches = re.findall(r"filename=\"(.+)\"", headers.get("Content-Disposition", ""))
    file_name = matches[0] if matches else None

    return ResponseInfo(status=status, file_size=file_size, file_name=file_name)


def get_download_path(
    dir_path: Optional[Path] = None,
    file_name: Optional[Union[str, Path]] = None,
    unique_key: Optional[str] = None,
) -> Path:
    """
    Determine the file path based on dir_path, file_name and/or data

    Args:
        dir_path:   An optional dir path to store the downloaded file. If no dir_path is given the system's temp dir
                    will be used. If omitted, the tempfolder is used.
        file_name:  An optional file name (or path relative to dir_path). If no file_name is given, a file name is
                    generated based on the unique key (e.g. an url)
        unique_key: A unique string that can be used to generate a filename (e.g. a url).
    """

    # If no specific download path was given, we need to generate a unique key (based on the given unique key)
    if file_name is None or unique_key is not None:
        if unique_key is None:
            raise ValueError("Supply a unique key in order to auto generate a download path.")

        sha256 = hashlib.sha256()
        sha256.update(unique_key.encode())
        unique_key = base64.b64encode(sha256.digest()).decode("ascii")
        unique_key = unique_key.replace("/", "_").replace("+", "-").rstrip("=")

        # If no file name was given, use the unique key as a file name
        if file_name is None:
            file_name = Path(f"{unique_key}.download")
        # Otherwise, use the unique key as a sub directory
        elif dir_path is None:
            dir_path = Path(tempfile.gettempdir()) / unique_key

    # If no dir_path is given, use the system's designated folder for temporary files.
    if dir_path is None:
        dir_path = Path(tempfile.gettempdir())

    # Combine the two paths
    assert file_name is not None
    file_path = (dir_path / file_name) if dir_path else Path(file_name)

    # If the file_path exists, it should be a file (not a dir)
    if file_path.exists() and not file_path.is_file():
        raise ValueError(f"Invalid file path: {file_path}")

    return file_path.resolve()
