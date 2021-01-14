# -*- coding: utf-8 -*-
"""
Tests for str() and repr() on RaggedArrays.
"""

import re

import numpy as np
import pytest

from rockhopper import RaggedArray, ragged_array

pytestmark = pytest.mark.order(2)

REPRs = [
    """\
RaggedArray.from_nested([
    [0, 1, 2, 3, 4],
    [5],
    [ 6,  7,  8,  9, 10, 11, 12, 13, 14],
    [15],
    [],
])""", """\
RaggedArray.from_nested([
    [],
    [[4., 5., 6., 7.]],
    [[ 8.,  9., 10., 11.],
     [12., 13., 14., 15.],
     [16., 17., 18., 19.]],
    [],
    [],
])"""
]


@pytest.mark.parametrize("repr_", REPRs)
def test_repr(repr_):
    self = eval(repr_)
    assert isinstance(self, RaggedArray)
    assert repr(self) == repr_


STRs = [
    """\
[[[0. 1. 2.]]
 [[3. 4. 5.]
  [6. 7. 8.]]
 [[ 9. 10. 11.]]
 []
 [[12. 13. 14.]
  [15. 16. 17.]]
 []
 []]"""
]


@pytest.mark.parametrize("str_", STRs)
def test_str(str_):
    nested = eval(re.sub(r"\s+", ",", re.sub(r"\[\s+", "[", str_)))
    self = RaggedArray.from_nested(nested)
    assert str(self) == str_


def test_long_repr():
    self = ragged_array(np.arange(1 << 12)[:, np.newaxis])
    assert repr(self) == """\
RaggedArray.from_nested([
    [0],
    [1],
    [2],
    ...,
    [4093],
    [4094],
    [4095],
])"""
