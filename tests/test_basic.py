# -*- coding: utf-8 -*-
"""
"""

import numpy as np

from rockhopper import RaggedArray


def test_implicit_bounds():
    flat = np.random.random(10)
    bounds = [0, 3, 8, 8, 10]

    self = RaggedArray(flat, bounds)
    assert self.flat is flat
    assert np.all(self.starts == bounds[:-1])
    assert np.all(self.ends == bounds[1:])


def test_explicit_bounds():
    flat = np.random.random(10)
    starts = [2, 4, 4, 9]
    ends = [4, 4, 8, 10]

    self = RaggedArray(flat, starts, ends)
    assert self.flat is flat
    assert np.all(self.starts == starts)
    assert np.all(self.ends == ends)
