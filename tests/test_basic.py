# -*- coding: utf-8 -*-
"""
"""

import sys
import collections

import numpy as np
import pytest

from rockhopper import RaggedArray, ragged_array

pytestmark = pytest.mark.order(0)


def test_implicit_bounds():
    flat = np.random.random(10)
    bounds = [0, 3, 8, 8, 10]

    self = RaggedArray(flat, bounds)
    assert self.flat is flat
    assert np.all(self.starts == bounds[:-1])
    assert np.all(self.ends == bounds[1:])

    _test_get_row(self)
    assert RaggedArray(flat, bounds, dtype=np.float32).dtype == np.float32


def test_explicit_bounds():
    flat = np.random.random(10)
    starts = [2, 4, 4, 9]
    ends = [4, 4, 8, 10]

    self = RaggedArray(flat, starts, ends)
    assert self.flat is flat
    assert np.all(self.starts == starts)
    assert np.all(self.ends == ends)

    _test_get_row(self)

    assert self.astype(np.float32).dtype == np.float32


def _test_get_row(self):
    assert len(self) == len(self.starts) == len(self.ends)

    for (i, row) in enumerate(self):
        # Test plain scalar get.
        assert np.array_equal(self[i], row)
        assert np.array_equal(self[i], self.flat[self.starts[i]:self.ends[i]])

        # Test slice.
        sliced = self[:i]
        assert len(sliced) == i
        if i > 0:
            assert np.array_equal(sliced[-1], self[i - 1])

        # Test array of indices.
        idx = np.random.randint(0, len(self), i)
        indexed = self[idx]
        assert all(map(np.array_equal, (self[i] for i in idx), indexed))


def test_from_lengths():
    flat = np.arange(10)
    self = RaggedArray.from_lengths(flat, [2, 3, 0, 4])
    assert self.flat is flat
    assert np.array_equal(self.starts, [0, 2, 5, 5])
    assert np.array_equal(self.ends, [2, 5, 5, 9])


NESTED = [[0, 1, 2], [3, 4], [], [5], [6, 7, 8, 9]]


@pytest.mark.parametrize("dtype", [None, int, np.float32])
def test_from_nested(dtype):
    self = RaggedArray.from_nested(NESTED, dtype=dtype)
    assert len(self) == len(NESTED)
    assert np.array_equal([len(i) for i in NESTED], self.ends - self.starts)
    assert all(map(np.array_equal, self, NESTED))
    if dtype is None:
        assert self.dtype == int
    else:
        assert self.dtype == dtype


def test_structured_from_nested():
    """Structured NumPy arrays need to be reminded to stay structured and not
    to simply cast everything to a *fits all* type (usually a string)."""
    dtype = [("foo", str, 3), ("bar", int)]

    self = ragged_array([
        [("abc", 3), ("efg", 5)],
        [("xyz", 9)],
    ], dtype=dtype)

    assert len(self) == 2
    assert self.dtype == dtype
    assert self[0]["foo"].tolist() == ["abc", "efg"]


def test_repacked():
    flat = np.random.random(10)
    starts, ends = zip([2, 4], [5, 5], [3, 8], [8, 10])

    self = RaggedArray(flat, starts, ends)
    packed = self.repacked()

    assert len(packed) == len(self)
    assert np.array_equal(packed.ends - packed.starts, self.ends - self.starts)
    assert all(map(np.array_equal, self, packed))
    assert np.array_equal(packed.starts[1:], packed.ends[:-1])
    assert packed.starts[0] == 0
    assert packed.ends[-1] == len(packed.flat)


@pytest.mark.parametrize("n", [10, 1, 0, 1000])
def test_rectangular(n):
    """Test :meth:`RaggedArray.to_rectangular_arrays()` on arrays of different
    sizes.
    """
    np.random.seed(0)
    self = RaggedArray(np.arange(n), np.sort(np.random.randint(0, n, n)))
    lengths = np.array([len(i) for i in self])

    out = self.to_rectangular_arrays()
    out_shapes = [i.shape for i in out]

    start = 0
    for (count, length) in out_shapes:
        assert np.all(lengths[start:start + count] == length)
        start += count

    if len(self):
        assert np.array_equal(self.repacked().flat,
                              np.concatenate(out, axis=None))
    else:
        assert len(out) == 0


@pytest.mark.parametrize("n", [10, 1, 0, 1000])
def test_sorted_rectangular(n):
    """Test :meth:`RaggedArray.to_rectangular_arrays(reorder=True)`."""
    np.random.seed(0)
    self = RaggedArray(np.arange(n), np.sort(np.random.randint(0, n, n)))
    args, out = self.to_rectangular_arrays(reorder=True)

    # The shapes of the arrays in ``out`` should be counts of rows in ``self``
    # with a given row length. ``out_shapes`` should be a list of
    # ``(number_of_rows_of_length, row_length)`` pairs, sorted in ascending
    # order of ``row_length``.
    out_shapes = [i.shape for i in out]
    # Check that the above is true.
    counts = collections.Counter(len(i) for i in self)
    assert [i[::-1] for i in sorted(counts.items())] == out_shapes

    if len(args):
        # The flattened data should have been reordered but otherwise preserved.
        assert np.array_equal(self[args].repacked().flat,
                              np.concatenate(out, axis=None))
    else:
        # ``np.concatenate()`` requires at least one input.
        assert out == []


def test_3d():
    self = RaggedArray.from_nested([
        [[0, 1, 2], [3, 4, 5]],
        [[6, 7, 8], [9, 10, 11]],
        [[12, 13, 14], [15, 16, 17], [18, 19, 20]],
        [],
    ])
    assert np.array_equal(self.flat, np.arange(21).reshape((7, 3)))
    assert len(self) == 4
    assert self.dtype == int
    assert self.itemshape == (3,)
    assert self.itemsize == 3 * self.dtype.itemsize
    assert self[-1].shape == (0, 3)

    # This array is already packed so `repacked` should be an exact copy.
    repacked = self.repacked()
    assert np.array_equal(self.flat, repacked.flat)
    assert np.array_equal(self.starts, repacked.starts)
    assert np.array_equal(self.ends, repacked.ends)

    cuboidals = self.to_rectangular_arrays()
    assert len(cuboidals) == 3
    assert cuboidals[0].shape == (2, 2, 3)
    assert cuboidals[1].shape == (1, 3, 3)
    assert cuboidals[2].shape == (1, 0, 3)
    flat = np.concatenate([i.reshape((-1, 3)) for i in cuboidals], axis=0)
    assert np.array_equal(flat, self.flat)


@pytest.mark.skipif(sys.maxsize < 1 << 32,
                    reason="Irrelevant on 32 bit platforms.")
def test_too_big():
    flat = np.empty(1 << 31, np.dtype([]))
    with pytest.raises(NotImplementedError, match="Flat lengths .*"):
        RaggedArray(flat, [])


EMPTY = []
SIMPLE = [[1, 2], [], [3], [4, 5, 6]]
COMPOUND = [[[1, 2]], [[3, 4], [5, 6]], []]


@pytest.mark.parametrize("nested", [EMPTY, SIMPLE, COMPOUND])
def test_to_list(nested):
    self = ragged_array(nested)
    assert self.tolist() == nested


def test_check():
    with pytest.raises(ValueError,
                       match=r".* lengths .* \(5\) .* \(6\) do not match"):
        RaggedArray(np.empty(10), np.arange(5), np.arange(6))

    with pytest.raises(ValueError,
                       match=r"Row 2, .* flat\[5\] .* flat\[3\], .* \(-2\)"):
        RaggedArray(np.empty(10), [0, 2, 5, 1], [1, 2, 3, 3])

    with pytest.raises(ValueError,
                       match=r"Row 1, .* flat\[5\] .* flat\[4\], .* \(-1\)"):
        RaggedArray(np.empty(10), [0, 5, 4, 6, 7])

    with pytest.raises(IndexError, match=r"starts\[2\] = -2 < 0"):
        RaggedArray(np.empty(10), [0, 1, -2, -3, 4], [1, 2, 3, 4, 5])

    with pytest.raises(IndexError, match=r"ends\[3\] = 14 >= len\(flat\) = 10"):
        RaggedArray(np.empty(10), [0, 1, 2, 3, 4], [1, 2, 3, 14, 5])


@pytest.mark.parametrize("in_place", [False, True])
def test_byteswap(in_place):
    self = RaggedArray.from_nested(SIMPLE, dtype=np.uint16)

    swapped = self.byteswap(inplace=in_place)
    assert (self is swapped) is in_place
    assert np.shares_memory(self.flat, swapped.flat) is in_place
    assert swapped.dtype == np.uint16
    assert swapped[0].tolist() == [0x0100, 0x0200]
