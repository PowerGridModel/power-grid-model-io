# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import structlog.testing

from power_grid_model_io.utils.zip import _get_only_item_in_dir, extract

from ...utils import MockTqdm, assert_log_exists

DATA_DIR = Path(__file__).parents[2] / "data" / "zip"
ZIP1 = DATA_DIR / "foo.zip"
ZIP2 = DATA_DIR / "foo-bar.zip"


@pytest.fixture()
def temp_dir():
    with tempfile.TemporaryDirectory() as tmp:
        yield Path(tmp).resolve()


@patch("power_grid_model_io.utils.download.tqdm")
def test_extract(mock_tqdm: MagicMock, temp_dir: Path):
    # Arrange
    src_file_path = temp_dir / "compressed.zip"
    dst_dir_path = temp_dir / "extracted"
    shutil.copyfile(ZIP2, src_file_path)
    mock_tqdm.side_effect = MockTqdm

    # Act
    extract_dir_path = extract(src_file_path=src_file_path, dst_dir_path=dst_dir_path)

    # Assert
    assert extract_dir_path == dst_dir_path
    assert (dst_dir_path / "foo.txt").is_file()
    assert (dst_dir_path / "folder/bar.txt").is_file()


@patch("power_grid_model_io.utils.download.tqdm")
def test_extract__auto_dir(mock_tqdm: MagicMock, temp_dir: Path):
    # Arrange
    src_file_path = temp_dir / "compressed.zip"
    shutil.copyfile(ZIP2, src_file_path)
    mock_tqdm.side_effect = MockTqdm

    # Act
    extract_dir_path = extract(src_file_path=src_file_path)

    # Assert
    assert extract_dir_path == temp_dir / "compressed"
    assert (temp_dir / "compressed" / "foo.txt").is_file()
    assert (temp_dir / "compressed" / "folder" / "bar.txt").is_file()


def test_extract__invalid_file_extension():
    # Act / Assert
    with pytest.raises(ValueError, match=r"Only files with \.zip extension are supported, got tempfile\.download"):
        extract(src_file_path=Path("/tmp/dir/tempfile.download"))


def test_extract__invalid_dst_dir(temp_dir: Path):
    # Arrange
    with open(temp_dir / "notadir.txt", "wb"):
        pass

    # Act / Assert
    with pytest.raises(NotADirectoryError, match=r"Destination dir .*notadir\.txt exists and is not a directory"):
        extract(src_file_path=Path("file.zip"), dst_dir_path=temp_dir / "notadir.txt")


@patch("power_grid_model_io.utils.download.tqdm")
def test_extract__file_exists(mock_tqdm: MagicMock, temp_dir: Path):
    # Arrange
    src_file_path = temp_dir / "compressed.zip"
    dst_dir_path = temp_dir / "extracted"
    shutil.copyfile(ZIP2, src_file_path)
    mock_tqdm.side_effect = MockTqdm

    dst_dir_path.mkdir()
    with open(dst_dir_path / "foo.txt", "wb") as fp:
        fp.write(b"\0")

    # Act / Assert
    with pytest.raises(FileExistsError, match=r"Destination file .*foo\.txt exists and is not empty"):
        extract(src_file_path=src_file_path, dst_dir_path=dst_dir_path)


@patch("power_grid_model_io.utils.download.tqdm")
def test_extract__skip_if_exists(mock_tqdm: MagicMock, temp_dir: Path):
    # Arrange
    src_file_path = temp_dir / "compressed.zip"
    dst_dir_path = temp_dir / "compressed"
    shutil.copyfile(ZIP2, src_file_path)
    mock_tqdm.side_effect = MockTqdm

    dst_dir_path.mkdir()
    with open(dst_dir_path / "foo.txt", "wb") as fp:
        fp.write(b"\0")

    # Act / Assert
    with structlog.testing.capture_logs() as capture:
        extract(src_file_path=src_file_path, dst_dir_path=dst_dir_path, skip_if_exists=True)
        assert_log_exists(
            capture, "debug", "Skip file extraction, destination file exists", dst_file_path=dst_dir_path / "foo.txt"
        )


@patch("power_grid_model_io.utils.download.tqdm")
def test_extract__return_subdir_path(mock_tqdm: MagicMock, temp_dir: Path):
    # Arrange
    src_file_path = temp_dir / "foo.zip"
    shutil.copyfile(ZIP1, src_file_path)
    mock_tqdm.side_effect = MockTqdm

    # Act
    extract_dir_path = extract(src_file_path=src_file_path)

    # Assert
    assert extract_dir_path == temp_dir / "foo" / "foo"
    assert (temp_dir / "foo" / "foo" / "foo.txt").is_file()


def test_get_only_item_in_dir__no_items(temp_dir):
    # Act / Assert
    assert _get_only_item_in_dir(temp_dir) == None


def test_get_only_item_in_dir__one_file(temp_dir):
    # Arrange
    with open(temp_dir / "file.txt", "wb"):
        pass

    # Act / Assert
    assert _get_only_item_in_dir(temp_dir) == temp_dir / "file.txt"


def test_get_only_item_in_dir__one_dir(temp_dir):
    # Arrange
    (temp_dir / "subdir").mkdir()

    # Act / Assert
    assert _get_only_item_in_dir(temp_dir) == temp_dir / "subdir"


def test_get_only_item_in_dir__two_files(temp_dir):
    # Arrange
    with open(temp_dir / "file_1.txt", "wb"):
        pass
    with open(temp_dir / "file_2.txt", "wb"):
        pass

    # Act / Assert
    assert _get_only_item_in_dir(temp_dir) == None
