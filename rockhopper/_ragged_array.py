# -*- coding: utf-8 -*-
"""
"""

import numpy as np


class RaggedArray(object):
    """A 2D array with rows of mixed lengths.

    A ragged array consists of three 1D arrays.

    *   :attr:`flat` contains the flattened contents. i.e. each row joined end
        end without any delimiters or information describing the shape.
    *   :attr:`starts` and :attr:`ends` determine the shape. Each integer value
        in these arrays is the start and stop of a :class:`slice` of
        :attr:`flat`. Each slice is a :class:`RaggedArray` row.

    A :class:`RaggedArray` is considered *packed* if the end of each row
    is the same as the start of the next row.

    """
    flat: np.ndarray
    starts: np.ndarray
    ends: np.ndarray

    def __init__(self, flat, starts, ends=None):
        """The default way to construct a :class:`RaggedArray` is explicitly
        from a :attr:`flat` contents array and either row :attr:`starts` and
        :attr:`ends` arrays or, more commonly, a *bounds* array.

        Args:
            flat:
                The contents of the array with no structure.
            starts:
                The index of **flat** where each row starts.
                Or if **ends** is unspecified, the start of each row and the
                end of the previous row.
            ends:
                The index of **flat** where each row ends.

        """
        self.flat = np.ascontiguousarray(flat)
        if ends is None:
            bounds = np.asarray(starts, dtype=np.intc, order="C")
            self.starts = bounds[:-1]
            self.ends = bounds[1:]
        else:
            self.starts = np.asarray(starts, dtype=np.intc, order="C")
            self.ends = np.asarray(ends, dtype=np.intc, order="C")
