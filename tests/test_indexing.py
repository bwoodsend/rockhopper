"""
Test all the various getitem/setitem flavours.
"""

import numpy as np
import pytest

from rockhopper import RaggedArray, RequestMeError
from tests import sliced

NESTED = [[1, 2, 3], [4, 5]]


def test_index_index():
    """Test ragged[number, number]"""
    self = RaggedArray.from_nested(NESTED)

    # Regular scalar lookup.
    assert self[0, 0] == 1
    assert self[0, 1] == 2
    assert self[(0, 1)] == 2
    assert self[1, -1] == 5
    assert self[-2, -3] == 1

    # The various index out of bounds errors.
    # Make sure that the right numbers are reported in the error messages.
    with pytest.raises(IndexError, match="10 .* 2"):
        # This is a regular NumPy exception.
        self[10, 0]
    with pytest.raises(IndexError, match="Index 4 .* row 0 .* size 3"):
        self[0, 4]
    with pytest.raises(IndexError, match="Index 3 .* row 1 .* size 2"):
        self[1, 3]
    with pytest.raises(IndexError, match="Index -3 .* row 1 .* size 2"):
        self[1, -3]

    # Bulk scalar lookup.
    assert self[[1, 0], [1, 2]].tolist() == [5, 3]
    assert self[[0, 1, 0], [2, 0, 2]].tolist() == [3, 4, 3]

    with pytest.raises(IndexError, match="Index 2 .* row 1 .* size 2"):
        self[[0, 1, 0], [1, 2, 4]]

    assert self[0, [1, 2, 0]].tolist() == [2, 3, 1]
    assert self[[1, 0], 0].tolist() == [4, 1]
    assert self[[[1], [0]], [[0, 1]]].tolist() == [[4, 5], [1, 2]]

    with pytest.raises(IndexError, match="Index 2 .* row 1 .* size 2"):
        self[[0, 1], 2]
    with pytest.raises(IndexError, match="2 .* axis 0 .* size 2"):
        # This is a regular NumPy exception.
        self[[0, 2], 1]


BIG_NESTED = [
    [0, 1],
    [2],
    [3, 4, 5, 6],
    [7],
    [8, 9, 10],
]


@pytest.mark.parametrize("rows, columns", [
    sliced[:, :],
    sliced[:, :0],
    sliced[:, :1],
    sliced[:, :2],
    sliced[:, :-1],
    sliced[:, 1:],
    sliced[:, 1:3],
    sliced[:, 1:-1],
    sliced[:, -2:-1],
    sliced[:, -1:-2],
])
def test_number_slice(rows, columns):
    """Test ragged[number, slice]"""
    self = RaggedArray.from_nested(BIG_NESTED)

    target = [i[columns] for i in BIG_NESTED[rows]]
    assert self[rows, columns].tolist() == target


def test_write_number_slice():
    self = RaggedArray.from_nested(BIG_NESTED)

    with pytest.raises(RequestMeError):
        self[:, :2] = 1


def test_slice_index():
    """Test ragged[slice, number]"""
    self = RaggedArray.from_nested(BIG_NESTED)

    assert self[:, 0].tolist() == [0, 2, 3, 7, 8]
    assert self[:, -1].tolist() == [1, 2, 6, 7, 10]

    assert self[:2, 0].tolist() == [0, 2]
    assert self[7:, 0].tolist() == []
    assert self[::2, 1].tolist() == [1, 4, 9]

    assert self[2::2, np.arange(2)].tolist() == [[3, 4], [8, 9]]
    assert self[2::2, np.arange(-1, 2)].tolist() == [[6, 3, 4], [10, 8, 9]]

    with pytest.raises(IndexError, match="Index -2 .* row 3 .* size 1"):
        self[2:, [0, -1, -2, 0]]


def test_write_slice_index():
    """Test ragged[slice, number] = x"""
    self = RaggedArray.from_nested(BIG_NESTED)

    self[:, 0] = [1, 2, 3, 4, 5]
    assert self[:, 0].tolist() == [1, 2, 3, 4, 5]
    assert self[:2, 0].tolist() == [1, 2]

    self[:2, -1] = 99
    assert self[0, -1] == self[1, -1] == 99

    # Neither of these should do anything because there is no row 7.
    self[7:, 0] = []
    self[7:, 0] = 0

    self[::2, 1] = [100, 101, 102]
    assert self[::2, 1].tolist() == [100, 101, 102]

    self[0::2, np.arange(-1, 2)] = [[51, 52, 53], [54, 55, 56], [57, 58, 59]]
    # self[0, 1] and self[0, -1] are the same so that cell gets written to
    # twice. It takes the most recently set value (53). All the rest are simply
    # what went in.
    assert self[::2, [-1, 0, 1]].tolist() == \
           [[53, 52, 53], [54, 55, 56], [57, 58, 59]]


def test_3d():
    self = RaggedArray.from_nested([
        [[0, 1, 2], [3, 4, 5]],
        [[6, 7, 8], [9, 10, 11]],
        [[12, 13, 14], [15, 16, 17], [18, 19, 20]],
        [],
    ])

    assert self[2, 1].tolist() == [15, 16, 17]
    assert self[2, 1, 2] == 17

    assert self[[2, 0], [1, 1]].tolist() == [[15, 16, 17], [3, 4, 5]]
    assert self[[2, 0], [1, 1], [0, 1]].tolist() == [15, 4]

    with pytest.raises(RequestMeError, match="Returning ragged .* from >2D"):
        self[:3, :2, 0]


def test_misc_exceptions():
    self = RaggedArray.from_nested(NESTED)

    with pytest.raises(RequestMeError, match="A stepped columns index"):
        self[2, ::2]

    with pytest.raises(IndexError, match="Too many indices .* 2 but 3 "):
        self[0, 0, 0]
