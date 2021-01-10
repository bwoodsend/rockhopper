# -*- coding: utf-8 -*-
"""
"""

import sys

import numpy as np
from cslug import CSlug, ptr, anchor, Header

BIG_ENDIAN = sys.byteorder == "big"
endians_header = Header(*anchor("src/endians.h", "src/endians.c"),
                        includes=["<stdbool.h>", '"_endian_typedefs.h"'])
slug = CSlug(anchor(
    "_slugs/ragged_array",
    "src/ragged_array.c",
    "src/ragged_array.h",
    "src/endians.c",
), headers=endians_header)  # yapf: disable


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

    def __init__(self, flat, starts, ends=None, dtype=None):
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
        self.flat = np.asarray(flat, dtype=dtype, order="C")
        if ends is None:
            bounds = np.asarray(starts, dtype=np.intc, order="C")
            self.starts = bounds[:-1]
            self.ends = bounds[1:]
        else:
            self.starts = np.asarray(starts, dtype=np.intc, order="C")
            self.ends = np.asarray(ends, dtype=np.intc, order="C")

        self._c_struct = slug.dll.RaggedArray(
            ptr(self.flat),
            self.flat.dtype.itemsize,
            len(self),
            ptr(self.starts),
            ptr(self.ends),
        )

    @property
    def dtype(self):
        """The data type of the contents of this array.

        Returns:
            numpy.dtype: :py:`self.flat.dtype`.

        """
        return self.flat.dtype

    def astype(self, dtype):
        """Cast the contents to a given **dtype**. Analogous to
        :meth:`numpy.ndarray.astype`.

        Args:
            dtype (numpy.dtype):
                Desired data type for the :attr:`flat` attribute.

        Returns:
            RaggedArray: A modified copy with :py:`copy.flat.dtype == dtype`.

        Only the :attr:`flat` property is cast - :attr:`starts` and :attr:`ends`
        remain unchanged.

        The :attr:`flat` attribute is a copy if :meth:`numpy.ndarray.astype`
        chooses to copy it. The :attr:`starts` and :attr:`ends` are never
        copied.

            >>> ragged = RaggedArray.from_nested([[1, 2], [3]], dtype=np.int32)
            >>> ragged.astype(np.int32).flat is ragged.flat
            False
            >>> ragged.astype(np.int16).starts is ragged.starts
            True

        """
        return type(self)(self.flat.astype(dtype), self.starts, self.ends)

    @classmethod
    def from_lengths(cls, flat, lengths, dtype=None):
        bounds = np.empty(len(lengths) + 1, dtype=np.intc)
        bounds[0] = 0
        np.cumsum(lengths, out=bounds[1:])
        return cls(flat, bounds, dtype=dtype)

    @classmethod
    def from_nested(cls, nested, dtype=None):
        flat = np.concatenate(nested)
        lengths = [len(i) for i in nested]
        return cls.from_lengths(flat, lengths, dtype=dtype)

    def __getitem__(self, item):
        if np.isscalar(item):
            return self.flat[self.starts[item]:self.ends[item]]
        return type(self)(self.flat, self.starts[item], self.ends[item])

    def __len__(self):
        return len(self.starts)

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def repacked(self):
        length = (self.ends - self.starts).sum()
        flat = np.empty((length,) + self.flat.shape[1:], self.flat.dtype)
        bounds = np.empty(len(self.starts) + 1, np.intc)
        new = type(self)(flat, bounds[:-1], bounds[1:])
        slug.dll.repack(self._c_struct._ptr, new._c_struct._ptr)
        return new

    def dumps(self, lengths_dtype=np.intc):
        """Serialise into a :class:`memoryview`.

        Args:
            lengths_dtype (numpy.dtype):
                Integer type.

        Returns:
            memoryview:
                Binary blob.

        The binary format is an undelimited sequence of ``(len(row), row)``
        pairs. A pure Python approximation would be::

            b"".join((len(row).tobytes() + row.tobytes() for row in ragged_array))

        The integer types of the row lengths can be controlled by the
        **lengths_dtype** parameter. To change the type or byteorder of the data
        itself, cast to that type with :meth:`astype` then call this function.

        """
        lengths_dtype = np.dtype(lengths_dtype)

        # --- Work out how many bytes the output will need. ---

        # The total length of the flat data. Note, `self.flat.size` would not be
        # a safe shortcut unless `self.repacked()` has been called 1st.
        length = (self.ends - self.starts).sum() * self.dtype.itemsize
        # And the lengths of the lengths...
        length += len(self) * lengths_dtype.itemsize

        # Allocate `length` bytes to write to. `numpy.empty()` seems to be one
        # of the only ways to create a lump of memory in Python without wasting
        # time initialising it.
        out = np.empty(length, dtype=np.byte)

        slug.dll.dump(self._c_struct._ptr, ptr(out), _2_power(lengths_dtype),
                      _big_endian(lengths_dtype))
        return out.data

    @classmethod
    def loads(cls, bin, rows=-1, dtype=None, lengths_dtype=np.intc):
        """Deserialize a ragged array. This is the reciprocal of :meth:`dumps`.

        Args:
            bin (bytes):
                Raw data to unpack.
            rows (int):
                Number of rows to parse. Defaults to :py:`-1` for unknown.
            dtype (numpy.dtype):
                Data type of the row contents in **bin**.
            lengths_dtype (numpy.dtype):
                Integer type of the row lengths in **bin**.
        Returns:
            RaggedArray:

        Raises:
            ValueError:
                If **bin** ends prematurely or in the middle of a row. This is
                indicative of either data corruption or, more likely, muddling
                of dtypes.

        """
        dtype = np.dtype(dtype)
        lengths_dtype = np.dtype(lengths_dtype)

        # We need to know how many rows there will be in this new ragged array
        # before creating and populating it.
        if rows == -1:
            # If it's not already known then it has to be counted.
            rows = slug.dll.count_rows(ptr(bin), len(bin),
                                       _2_power(lengths_dtype),
                                       _big_endian(lengths_dtype),
                                       dtype.itemsize)
            if rows == -1:
                # `count_rows()` returns -1 on error.
                raise ValueError(
                    "Raw `bin` data ended mid way through a row. Either this "
                    "data is corrupt or the dtype(s) given are incorrect.")

            # Run again with known number of `rows`.
            return cls.loads(bin, rows, dtype, lengths_dtype)

        items = (len(bin) - rows * lengths_dtype.itemsize) // dtype.itemsize

        self = cls(np.empty(items, dtype=dtype), np.empty(rows + 1, np.intc))

        slug.dll.load(self._c_struct._ptr, ptr(bin), len(bin),
                      _2_power(lengths_dtype), _big_endian(lengths_dtype),
                      dtype.itemsize)

        return self

    def _rectangular_slice(self, start, end):
        """Slice ``self`` but convert the output to a regular rectangular array.

        This requires that this array is packed, hence its being private.
        """
        width = self.ends[start] - self.starts[start]

        if end >= len(self):
            flat = self.flat[self.starts[start]:]
            return flat.reshape((len(self) - start, width))

        flat = self.flat[self.starts[start]:self.starts[end]]
        return flat.reshape((end - start, width))

    def to_rectangular_arrays(self, reorder=False):
        """Convert to a :class:`list` of regular :class:`numpy.ndarray`\\ s.

        Args:
            reorder (bool):
                If true, pre-sort into order of ascending lengths to minimise
                divisions needed. Use if the row order is unimportant.

        Returns:
            Union[tuple, list]:
                list[numpy.ndarray]:
                    If **reorder** is false.
                numpy.ndarray, list[numpy.ndarray]:
                    If **reorder** is true. The first argument is the args (from
                    :func:`numpy.argsort`) used to pre-sort.

        The :class:`RaggedArray` is divided into chunks of consecutive rows
        which have the same length. Each chunk is then converted to a plain 2D
        :class:`numpy.ndarray`. These 2D arrays are returned in a :class:`list`.
        ::

            >>> ragged_array([
            ...     [1, 2],
            ...     [3, 4],
            ...     [5, 6, 7],
            ...     [8, 9, 10],
            ... ]).to_rectangular_arrays()
            [array([[1, 2], [3, 4]]), array([[ 5,  6,  7], [ 8,  9, 10]])]

        """
        if reorder:
            args = np.argsort(self.ends - self.starts)
            return args, self[args].to_rectangular_arrays()

        # The empty case requires special handling or it hits index errors
        # further on.
        if len(self) == 0:
            return []

        # This function uses slices on raw ``self.flat`` and thereby assumes
        # that consecutive rows are consecutive in ``self.flat``. To enforce
        # this case:
        self = self.repacked()

        lengths = self.ends - self.starts

        out = []
        start = 0
        # For every row number that isn't the same length as its next row:
        for end in np.nonzero(lengths[1:] != lengths[:-1])[0]:
            end += 1
            # slice from the last slice end to this one.
            out.append(self._rectangular_slice(start, end))
            start = end

        # The above catches everything before a change in row length but not the
        # final chunk after the last change. Add it.
        out.append(self._rectangular_slice(start, len(self)))

        return out


def _2_power(dtype):
    """Convert an integer dtype to an enumerate used throughout the C code."""
    # Functionally this is equivalent to ``int(math.log2(dtype.itemsize))``.
    itemsize = np.dtype(dtype).itemsize
    return next(i for i in range(8) if (1 << i) == itemsize)


def _big_endian(dtype):
    """Is **dtype** bit endian?"""
    byteorder = np.dtype(dtype).byteorder
    if byteorder == "<":
        return False
    if byteorder == ">":
        return True
    # `byteorder` can also be '=' for native (`sys.endian == "big"`) or "|" for
    # not applicable (for string types - which we shouldn't need anyway - or
    # single byte types).
    return BIG_ENDIAN
