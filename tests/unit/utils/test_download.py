# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
import tempfile
from collections import namedtuple
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import pytest
import structlog.testing

from power_grid_model_io.utils.download import (
    DownloadProgressHook,
    ResponseInfo,
    download,
    download_and_extract,
    get_download_path,
    get_response_info,
)

from ...utils import assert_log_exists

Response = namedtuple("Response", ["status", "headers"])

# The base64 representation of the sha256 hash of "foo" is LCa0a2j/xo/5m0U8HTBBNBNCLXBkg7+g+YpeiGJm564=
# The / and + will be replaced with a _ and - character and the trailing = character(s) will be removed.
FOO_KEY = "LCa0a2j_xo_5m0U8HTBBNBNCLXBkg7-g-YpeiGJm564"

TEMP_DIR = Path(tempfile.gettempdir()).resolve()


@pytest.fixture()
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp).resolve()


def make_file(file_path: Path, file_size: int = 0):
    with open(file_path, "wb") as fp:
        fp.write(b"\0" * file_size)


def test_progress_hook():
    # Arrange
    progress_bar = MagicMock()
    progress_bar.total = None
    hook = DownloadProgressHook(progress_bar)

    # Act (block_num, block_size, file_size)
    hook(2, 10, 0)
    assert progress_bar.total is None  # total is not updated

    hook(3, 10, 123)
    assert progress_bar.total == 123  # total is updated

    hook(6, 10, 123)

    # Assert
    assert progress_bar.update.call_args_list[0].args == (20,)
    assert progress_bar.update.call_args_list[1].args == (10,)
    assert progress_bar.update.call_args_list[2].args == (30,)


@patch("power_grid_model_io.utils.download.extract")
@patch("power_grid_model_io.utils.download.download")
def test_download_and_extract__paths(mock_download: MagicMock, mock_extract: MagicMock, temp_dir: Path):
    # Arrange
    url = MagicMock()
    dir_path = MagicMock()
    file_path = MagicMock()
    src_file_path = temp_dir / "data.zip"
    dst_dir_path = temp_dir / "data"
    extract_dir_path = MagicMock()

    mock_download.return_value = src_file_path
    mock_extract.return_value = extract_dir_path

    # Act
    result = download_and_extract(url=url, dir_path=dir_path, file_name=file_path)

    # Assert
    mock_download.assert_called_once_with(url=url, file_name=file_path, dir_path=dir_path, overwrite=False)
    mock_extract.assert_called_once_with(src_file_path=src_file_path, dst_dir_path=dst_dir_path, skip_if_exists=True)
    assert result == extract_dir_path


@patch("power_grid_model_io.utils.download.extract")
@patch("power_grid_model_io.utils.download.download")
def test_download_and_extract__no_paths(mock_download: MagicMock, mock_extract: MagicMock, temp_dir: Path):
    # Arrange
    url = MagicMock()
    src_file_path = temp_dir / "data.zip"
    dst_dir_path = temp_dir / "data"

    mock_download.return_value = src_file_path

    # Act
    download_and_extract(url=url)

    # Assert
    mock_download.assert_called_once_with(url=url, file_name=None, dir_path=None, overwrite=False)
    mock_extract.assert_called_once_with(src_file_path=src_file_path, dst_dir_path=dst_dir_path, skip_if_exists=True)


@patch("power_grid_model_io.utils.download.extract", new=MagicMock)
@patch("power_grid_model_io.utils.download.download")
def test_download_and_extract__overwrite(mock_download: MagicMock, temp_dir: Path):
    # Arrange
    src_file_path = temp_dir / "data.zip"
    mock_download.return_value = src_file_path

    dst_dir_path = temp_dir / "data"
    dst_dir_path.mkdir()

    # Act / Assert
    download_and_extract(url=MagicMock(), overwrite=False)
    assert dst_dir_path.is_dir()

    # Act / Assert (dir does exist, overwrite = True)
    download_and_extract(url=MagicMock(), overwrite=True)
    assert not dst_dir_path.exists()


@patch("power_grid_model_io.utils.download.DownloadProgressHook")
@patch("power_grid_model_io.utils.download.request.urlretrieve")
@patch("power_grid_model_io.utils.download.tqdm")
@patch("power_grid_model_io.utils.download.get_download_path")
@patch("power_grid_model_io.utils.download.get_response_info")
def test_download(
    mock_info: MagicMock,
    mock_download_path: MagicMock,
    mock_tqdm: MagicMock,
    mock_urlretrieve: MagicMock,
    mock_hook: MagicMock,
    temp_dir: Path,
):
    # Arrange
    url = "https://www.source.com"
    dir_path = temp_dir / "data"
    file_path = temp_dir / "data.zip"
    temp_file = temp_dir / "data.download"
    download_path = temp_dir / "data.zip"

    def urlretrieve(*_args, **_kwargs):
        make_file(temp_file, 100)
        return temp_file, None

    mock_info.return_value = ResponseInfo(status=200, file_size=100, file_name="remote.zip")
    mock_download_path.return_value = download_path
    mock_urlretrieve.side_effect = urlretrieve

    # Act / Assert
    with structlog.testing.capture_logs() as capture:
        result = download(url=url, file_name=file_path, dir_path=dir_path)
        assert_log_exists(capture, "debug", "Downloading file")

    # Assert
    mock_download_path.assert_called_once_with(
        dir_path=dir_path, file_name=file_path, unique_key="https://www.source.com"
    )
    mock_tqdm.assert_called_once()
    mock_hook.assert_called_once_with(mock_tqdm.return_value.__enter__.return_value)
    mock_urlretrieve.assert_called_once_with(url, reporthook=mock_hook.return_value)
    assert result == file_path
    assert result.is_file()
    assert result.stat().st_size == 100


@patch("power_grid_model_io.utils.download.DownloadProgressHook", new=MagicMock())
@patch("power_grid_model_io.utils.download.request.urlretrieve")
@patch("power_grid_model_io.utils.download.tqdm", new=MagicMock())
@patch("power_grid_model_io.utils.download.get_download_path")
@patch("power_grid_model_io.utils.download.get_response_info")
def test_download__auto_file_name(
    mock_info: MagicMock, mock_download_path: MagicMock, mock_urlretrieve: MagicMock, temp_dir: Path
):
    # Arrange
    temp_file = temp_dir / "data.download"
    download_path = temp_dir / "data.zip"

    def urlretrieve(*_args, **_kwargs):
        make_file(temp_file, 100)
        return temp_file, None

    mock_info.return_value = ResponseInfo(status=200, file_size=None, file_name="remote.zip")
    mock_download_path.return_value = download_path
    mock_urlretrieve.side_effect = urlretrieve

    # Act
    download(url=MagicMock())

    # Assert
    mock_download_path.assert_called_once_with(dir_path=None, file_name="remote.zip", unique_key=ANY)


@patch("power_grid_model_io.utils.download.DownloadProgressHook", new=MagicMock())
@patch("power_grid_model_io.utils.download.request.urlretrieve")
@patch("power_grid_model_io.utils.download.tqdm", new=MagicMock())
@patch("power_grid_model_io.utils.download.get_download_path")
@patch("power_grid_model_io.utils.download.get_response_info")
def test_download__empty_file(
    mock_info: MagicMock, mock_download_path: MagicMock, mock_urlretrieve: MagicMock, temp_dir: Path
):
    # Arrange
    temp_file = temp_dir / "data.download"
    download_path = temp_dir / "data.zip"

    def urlretrieve(*_args, **_kwargs):
        with open(temp_file, "wb"):
            pass
        return temp_file, None

    mock_info.return_value = ResponseInfo(status=200, file_size=None, file_name="remote.zip")
    mock_download_path.return_value = download_path
    mock_urlretrieve.side_effect = urlretrieve

    # Act / Assert
    with structlog.testing.capture_logs() as capture:
        download(url=MagicMock())
        assert_log_exists(capture, "debug", "Downloading file")
        assert_log_exists(capture, "warning", "Downloaded an empty file")


@patch("power_grid_model_io.utils.download.DownloadProgressHook", new=MagicMock())
@patch("power_grid_model_io.utils.download.request.urlretrieve")
@patch("power_grid_model_io.utils.download.tqdm", new=MagicMock())
@patch("power_grid_model_io.utils.download.get_download_path")
@patch("power_grid_model_io.utils.download.get_response_info")
def test_download__skip_existing_file(
    mock_info: MagicMock, mock_download_path: MagicMock, mock_urlretrieve: MagicMock, temp_dir: Path
):
    # Arrange
    download_path = temp_dir / "data.zip"
    make_file(download_path, 100)

    mock_info.return_value = ResponseInfo(status=200, file_size=100, file_name="remote.zip")
    mock_download_path.return_value = download_path

    # Act
    with structlog.testing.capture_logs() as capture:
        download(url=MagicMock())
        assert_log_exists(capture, "debug", "Skip downloading existing file")

    # Assert
    mock_urlretrieve.assert_not_called()


@patch("power_grid_model_io.utils.download.DownloadProgressHook", new=MagicMock())
@patch("power_grid_model_io.utils.download.request.urlretrieve")
@patch("power_grid_model_io.utils.download.tqdm", new=MagicMock())
@patch("power_grid_model_io.utils.download.get_download_path")
@patch("power_grid_model_io.utils.download.get_response_info")
def test_download__update_file(
    mock_info: MagicMock, mock_download_path: MagicMock, mock_urlretrieve: MagicMock, temp_dir: Path
):
    # Arrange
    temp_file = temp_dir / "data.download"
    download_path = temp_dir / "data.zip"
    make_file(download_path, 100)

    def urlretrieve(*_args, **_kwargs):
        make_file(temp_file, 101)
        return temp_file, None

    mock_info.return_value = ResponseInfo(status=200, file_size=101, file_name="remote.zip")
    mock_download_path.return_value = download_path
    mock_urlretrieve.side_effect = urlretrieve

    # Act / Assert
    with structlog.testing.capture_logs() as capture:
        result = download(url=MagicMock())
        assert_log_exists(capture, "debug", "Re-downloading existing file, because the size has changed")

    # Assert
    assert result == download_path
    assert result.is_file()
    assert result.stat().st_size == 101


@patch("power_grid_model_io.utils.download.DownloadProgressHook", new=MagicMock())
@patch("power_grid_model_io.utils.download.request.urlretrieve")
@patch("power_grid_model_io.utils.download.tqdm", new=MagicMock())
@patch("power_grid_model_io.utils.download.get_download_path")
@patch("power_grid_model_io.utils.download.get_response_info")
def test_download__overwrite(
    mock_info: MagicMock, mock_download_path: MagicMock, mock_urlretrieve: MagicMock, temp_dir: Path
):
    # Arrange
    temp_file = temp_dir / "data.download"
    download_path = temp_dir / "data.zip"
    make_file(download_path, 100)

    def urlretrieve(*_args, **_kwargs):
        make_file(temp_file, 100)
        return temp_file, None

    mock_info.return_value = ResponseInfo(status=200, file_size=100, file_name="remote.zip")
    mock_download_path.return_value = download_path
    mock_urlretrieve.side_effect = urlretrieve

    # Act / Assert
    with structlog.testing.capture_logs() as capture:
        result = download(url=MagicMock(), overwrite=True)
        assert_log_exists(capture, "debug", "Forced re-downloading existing file")

    # Assert
    assert result == download_path
    assert result.is_file()
    assert result.stat().st_size == 100


@patch("power_grid_model_io.utils.download.get_response_info")
def test_download__status_error(mock_info: MagicMock):
    # Arrange
    mock_info.return_value = ResponseInfo(status=404, file_size=None, file_name=None)

    # Act / Assert
    with pytest.raises(IOError, match=r"Could not download from URL, status=404"):
        download(url=MagicMock())


@patch("power_grid_model_io.utils.download.request.urlopen")
def test_get_response_info(mock_urlopen):
    # Arrange
    headers = {"Content-Length": "456", "Content-Disposition": 'form-data; name="ZipFile"; filename="filename.zip"'}
    mock_urlopen.return_value.__enter__.return_value = Response(status=123, headers=headers)

    # Act / Assert
    assert get_response_info("") == ResponseInfo(status=123, file_size=456, file_name="filename.zip")


@patch("power_grid_model_io.utils.download.request.urlopen")
def test_get_response_info__no_file_name(mock_urlopen):
    # Arrange
    headers = {"Content-Length": "456", "Content-Disposition": 'form-data; name="ZipFile"'}
    mock_urlopen.return_value.__enter__.return_value = Response(status=123, headers=headers)

    # Act / Assert
    assert get_response_info("") == ResponseInfo(status=123, file_size=456, file_name=None)


@patch("power_grid_model_io.utils.download.request.urlopen")
def test_get_response_info__no_disposition(mock_urlopen):
    # Arrange
    headers = {"Content-Length": "456"}
    Context = namedtuple("Context", ["status", "headers"])
    mock_urlopen.return_value.__enter__.return_value = Context(status=123, headers=headers)

    # Act / Assert
    assert get_response_info("") == ResponseInfo(status=123, file_size=456, file_name=None)


@patch("power_grid_model_io.utils.download.request.urlopen")
def test_get_response_info__no_length(mock_urlopen):
    # Arrange
    headers = {"Content-Disposition": 'form-data; name="ZipFile"; filename="filename.zip"'}
    mock_urlopen.return_value.__enter__.return_value = Response(status=123, headers=headers)

    # Act / Assert
    assert get_response_info("") == ResponseInfo(status=123, file_size=None, file_name="filename.zip")


def test_get_download_path(temp_dir: Path):
    # Act
    path = get_download_path(dir_path=temp_dir, file_name="file_name.zip")

    # Assert
    assert path == temp_dir / "file_name.zip"


def test_get_download_path__ignore_unique_key(temp_dir: Path):
    # Act
    path = get_download_path(dir_path=temp_dir, file_name="file_name.zip", unique_key="foo")

    # Assert
    assert path == temp_dir / "file_name.zip"


def test_get_download_path__temp_dir():
    # Act
    path = get_download_path(file_name="file_name.zip")

    # Assert
    assert path == TEMP_DIR / "file_name.zip"


def test_get_download_path__auto_dir():
    # Act
    path = get_download_path(file_name="file_name.zip", unique_key="foo")

    # Assert
    assert path == TEMP_DIR / FOO_KEY / "file_name.zip"


def test_get_download_path__auto_file_name(temp_dir: Path):
    # Arrange
    expected_file_name = f"{FOO_KEY}.download"

    # Act
    path = get_download_path(dir_path=temp_dir, unique_key="foo")

    # Assert
    assert path == temp_dir / expected_file_name


def test_get_download_path__missing_data(temp_dir: Path):
    # Act / Assert
    with pytest.raises(ValueError, match=r"Supply a unique key in order to auto generate a download path\."):
        get_download_path(dir_path=temp_dir)


def test_get_download_path__invalid_file_path(temp_dir: Path):
    # Arrange
    (temp_dir / "download").mkdir()

    # Act / Assert
    with pytest.raises(ValueError, match=r"Invalid file path:"):
        get_download_path(dir_path=temp_dir, file_name="download")
