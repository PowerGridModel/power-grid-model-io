# SPDX-FileCopyrightText: Contributors to the Power Grid Model project <powergridmodel@lfenergy.org>
#
# SPDX-License-Identifier: MPL-2.0
import pytest

from power_grid_model_io.utils.dict import merge_dicts


def test_merge_dicts__empty():
    # Act
    merged = merge_dicts()

    # Assert
    assert merged == {}


def test_merge_dicts__one():
    # Arrange
    foo = {"a": 1, "b": 2}

    # Act
    merged = merge_dicts(foo)
    foo["a"] = 0

    # Assert
    assert merged == {"a": 1, "b": 2}


def test_merge_dicts__two():
    # Arrange
    foo = {"a": 1, "b": 2}
    bar = {"c": 3, "d": 4}

    # Act
    merged = merge_dicts(foo, bar)
    foo["a"] = 0
    bar["c"] = 0

    # Assert
    assert merged == {"a": 1, "b": 2, "c": 3, "d": 4}


def test_merge_dicts__three():
    # Arrange
    foo = {"a": 1, "b": 2}
    bar = {"c": 3, "d": 4}
    baz = {"e": 5, "f": 6}

    # Act
    merged = merge_dicts(foo, bar, baz)
    foo["a"] = 0
    bar["c"] = 0
    baz["e"] = 0

    # Assert
    assert merged == {"a": 1, "b": 2, "c": 3, "d": 4, "e": 5, "f": 6}


def test_merge_dicts__duplicate():
    # Arrange
    foo = {"a": 1, "b": 2}
    bar = {"b": 2, "c": 3}

    # Act
    merged = merge_dicts(foo, bar)

    # Assert
    assert merged == {"a": 1, "b": 2, "c": 3}


def test_merge_dicts__conflict():
    # Arrange
    foo = {"a": 1, "b": 2}
    bar = {"b": 3, "c": 4}

    # Act / Assert
    with pytest.raises(KeyError, match=r"'b'"):
        merge_dicts(foo, bar)


def test_merge_dicts__recursive():
    # Arrange
    foo = {"a": {"b": {"c": 1}}}
    bar = {"a": {"b": {"d": 2}}}

    # Act
    merged = merge_dicts(foo, bar)

    # Assert
    assert merged == {"a": {"b": {"c": 1, "d": 2}}}
