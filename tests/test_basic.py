# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import pytest

from rockhopper import RaggedArray


def test_implicit_bounds():
    flat = np.random.random(10)
    bounds = [0, 3, 8, 8, 10]

    self = RaggedArray(flat, bounds)
    assert self.flat is flat
    assert np.all(self.starts == bounds[:-1])
    assert np.all(self.ends == bounds[1:])

    _test_get_item(self)
    assert RaggedArray(flat, bounds, dtype=np.float32).dtype == np.float32


def test_explicit_bounds():
    flat = np.random.random(10)
    starts = [2, 4, 4, 9]
    ends = [4, 4, 8, 10]

    self = RaggedArray(flat, starts, ends)
    assert self.flat is flat
    assert np.all(self.starts == starts)
    assert np.all(self.ends == ends)

    _test_get_item(self)

    assert self.astype(np.float32).dtype == np.float32


def _test_get_item(self):
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
    assert self.dtype == dtype
