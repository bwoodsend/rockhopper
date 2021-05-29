# -*- coding: utf-8 -*-
"""
"""

import sys

import numpy as np
from cslug import CSlug, ptr, anchor, Header

NUMPY_REPR = False

BIG_ENDIAN = sys.byteorder == "big"
endians_header = Header(*anchor("src/endians.h", "src/endians.c"),
                        includes=["<stdbool.h>", '"_endian_typedefs.h"'])
slug = CSlug(anchor(
    "_slugs/ragged_array",
    "src/ragged_array.c",
    "src/ragged_array.h",
    "src/endians.c",
), headers=endians_header)  # yapf: disable


def prod(iterable):
    """Equivalent to :func:`math.prod` introduced in Python 3.8. """
    out = 1
    for i in iterable:
        out = out * i
    return out


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
            dtype:
                The :class:`numpy.dtype` of the array. Usually this can be
                inferred from **flat** and is therefore not required to be set
                explicitly. To indicate that multiple scalars should be
                considered as one item, use a :class:`tuple` dtype.

        .. seealso::

            Explicit construction is rarely the most convenient way to build a
            :class:`RaggedArray`.
            See :meth:`from_nested` to construct from lists of lists.
            Or :meth:`from_lengths` to construct from flat data and row lengths.
            Or :meth:`group_by` to specify the row number explicitly for each
            item.

        Examples:

            Assuming the setup code::

                import numpy as np
                from rockhopper import RaggedArray

                flat = np.arange(10)

            ::

                >>> bounds = [0, 4, 7, 10]
                >>> RaggedArray(flat, bounds)
                RaggedArray.from_nested([
                    [0, 1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9],
                ])

            The **bounds** need not start at the beginning and end and the end.
            Note however that the leading and trailing items in **flat** are not
            represented in the repr. ::

                >>> bounds = [2, 4, 4, 5, 9]
                >>> RaggedArray(flat, bounds)
                RaggedArray.from_nested([
                    [2, 3],
                    [],
                    [4],
                    [5, 6, 7, 8],
                ])

            To be able to have gaps between rows or overlapping rows set both
            **starts** and **ends**. ::

                >>> starts = [0, 3, 1]
                >>> ends = [6, 6, 5]
                >>> RaggedArray(flat, starts, ends)
                RaggedArray.from_nested([
                    [0, 1, 2, 3, 4, 5],  # flat[0:6]
                    [3, 4, 5],           # flat[3:6]
                    [1, 2, 3, 4],        # flat[1:5]
                ])

            This form is typically not very useful but is given more to explain
            how the :class:`RaggedArray` works internally.
            Copy-less slicing uses this form heavily.

        """
        self.flat = np.asarray(flat, dtype=dtype, order="C")

        if len(self.flat) >= (1 << 31):  # pragma: 64bit
            # Supporting large arrays would require promoting all ints in the C
            # code to int64_t. Given that it takes at least 2GB of memory to get
            # an array this big, I doubt that this would be useful but I could
            # be wrong...
            raise NotImplementedError(
                "Flat lengths >= 2^31  are disabled at compile time to save on "
                "memory. If you genuinely need arrays this large then raise an "
                "issue at https://github.com/bwoodsend/rockhopper/issues/new")

        if ends is None:
            bounds = np.asarray(starts, dtype=np.intc, order="C")
            self.starts = bounds[:-1]
            self.ends = bounds[1:]
        else:
            self.starts = np.asarray(starts, dtype=np.intc, order="C")
            self.ends = np.asarray(ends, dtype=np.intc, order="C")

        self._c_struct = slug.dll.RaggedArray(
            ptr(self.flat),
            self.itemsize,
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

    @property
    def itemshape(self):
        """The shape of an individual element from :attr:`flat`.

        Returns:
            tuple: :py:`self.flat.shape[1:]`.

        Assuming :attr:`flat` is not empty, this is equivalent to
        :py:`self.flat[0].shape`. For a 2D ragged array, this is always simply
        :py:`()`.

        """
        return self.flat.shape[1:]

    @property
    def itemsize(self):
        """The size in bytes of an individual element from :attr:`flat`.

        Returns:
            int: Size of one element.

        Assuming :attr:`flat` is not empty, this is equivalent to
        :py:`len(self.flat[0].tobytes()`.

        """
        return prod(self.itemshape) * self.dtype.itemsize

    def astype(self, dtype):
        """Cast the contents to a given **dtype**. Analogous to
        :meth:`numpy.ndarray.astype`.

        Args:
            dtype (Union[numpy.dtype, Type[numpy.generic]]):
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
        flat = np.concatenate([i for i in nested if len(i)] or [[]])
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

    def _to_string(self, prefix, separator):
        """Convert to :class:`str`. A loose ragged equivalent of
        :func:`numpy.array2string()`.

        Args:
            prefix (str):
                How far to indent. See the **prefix** option for
                :func:`numpy.array2string()`.
            separator (str):
                The deliminator to be put between elements.

        Returns:
            str: Something stringy.

        """
        # TODO: Maybe expand and make this method public.
        _str = lambda x: np.array2string(x, prefix=prefix, separator=separator)

        if len(self) > np.get_printoptions()['threshold']:
            # Very long arrays should be summarised as [a, b, c, ..., x, y, z].
            edge_items = np.get_printoptions()["edgeitems"]

            rows = [_str(i) for i in self[:edge_items]]
            rows.append("...")
            rows += [_str(i) for i in self[-edge_items:]]

        else:
            rows = [_str(i) for i in self]

        # A downside of doing everything per row is that each row gets formatted
        # differently. NumPy don't expose any of their fancy dragon4 algorithm
        # functionality for choosing format options so I don't see any practical
        # way of changing this.

        return (separator.rstrip() + "\n" + " " * len(prefix)).join(rows)

    def __repr__(self):
        prefix = type(self).__name__ + ".from_nested("

        # I might make this a proper option in future.
        if NUMPY_REPR:  # pragma: no cover
            # Old school NumPy style formatting.
            return prefix + "[" + self._to_string(prefix + "[", ", ") + "])"

        # More trendy trailing comma formatting for `black` fanatics.
        return prefix + "[\n    " + self._to_string("    ", ", ") + ",\n])"

    def __str__(self):
        return "[" + self._to_string(" ", " ") + "]"

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
            lengths_dtype (Union[numpy.dtype, Type[numpy.generic]]):
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
        length = (self.ends - self.starts).sum() * self.itemsize
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
            dtype (Union[numpy.dtype, Type[numpy.generic]]):
                Data type of the row contents in **bin**.
            lengths_dtype (Union[numpy.dtype, Type[numpy.generic]]):
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

        free = len(bin) - rows * lengths_dtype.itemsize
        items = free // dtype.itemsize
        if items < 0:
            raise ValueError(
                f"With `bin` of length {len(bin)}, {rows} rows of "
                f"{lengths_dtype.itemsize} byte lengths leaves {free} bytes "
                f"for the flat data. Perhaps your data types are wrong?")

        self = cls(np.empty(items, dtype=dtype), np.empty(rows + 1, np.intc))

        _rows = slug.dll.load(self._c_struct._ptr, ptr(bin), len(bin), rows,
                              _2_power(lengths_dtype),
                              _big_endian(lengths_dtype), dtype.itemsize)
        if _rows < rows:
            raise ValueError(
                f"Raw `bin` data ended too soon. "
                f"Only {_rows} out of the requested {rows} rows were read. "
                f"Either this data is corrupt or the dtype(s) given are "
                "incorrect.")

        return self

    def _rectangular_slice(self, start, end):
        """Slice ``self`` but convert the output to a regular rectangular array.

        This requires that this array is packed, hence its being private.
        """
        width = self.ends[start] - self.starts[start]

        if end >= len(self):
            flat = self.flat[self.starts[start]:]
            return flat.reshape((len(self) - start, width) + self.itemshape)

        flat = self.flat[self.starts[start]:self.starts[end]]
        return flat.reshape((end - start, width) + self.itemshape)

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

    @classmethod
    def group_by(cls, data, ids, id_max=None, check_ids=True):
        """Group **data** by **ids**.

        Args:
            data (numpy.ndarray):
                Arbitrary values to be grouped. **data** can be of any dtype and
                be multidimensional.
            ids (numpy.ndarray):
                Integer array with the same dimensions as **data**.
            id_max (int):
                :py:`np.max(ids) + 1`. If already known, providing this value
                prevents it from being redundantly recalculated.

        Returns:
            RaggedArray:

        For each value in **data**, its corresponding ID in **ids** determines
        in which row the data value is placed. The order of data within rows is
        consistent with the order the appear in **data**.

        This method is similar to :meth:`pandas.DataFrame.groupby`. However, it
        will not uniquify and enumerate the property to group by.

        """
        # Just run ``groups_by()`` but with only one ``datas``.
        return next(cls.groups_by(ids, data, id_max=id_max))

    @classmethod
    def groups_by(cls, ids, *datas, id_max=None, check_ids=True):
        """Group each data from **datas** by **ids**.

        This function is equivalent to, but faster than, calling
        :meth:`group_by` multiple times with the same **ids**.
        """
        # Type normalisation and sanity checks.
        ids = np.asarray(ids)
        datas = [np.asarray(i) for i in datas]
        if id_max is None:
            id_max = np.max(ids) + 1
        elif check_ids and np.any(ids >= id_max):
            max = ids.argmax()
            raise IndexError(f"All ids must be < id_max but "
                             f"ids[{max}] = {ids[max]} >= {id_max}.")
        if check_ids and np.any(ids < 0):
            min = ids.argmin()
            raise IndexError(
                f"All ids must be >= 0 but ids[{min}] = {ids[min]}.")

        counts, sub_ids = sub_enumerate(ids, id_max)

        # The ``counts`` determine the lengths of each row should.
        # From there we can work out the start and end point for each row.
        bounds = np.empty(id_max + 1, np.intc)
        counts.cumsum(out=bounds[1:])
        bounds[0] = 0

        # The ``sub_ids`` are the position along the row for each element.
        # Without ``sub_ids``, elements from the same group will all write to
        # the beginning of their row, and thus overwrite each other.
        unique_ids = bounds[ids] + sub_ids
        # ``unique_ids`` should contain exactly one of each element in
        # ``range(len(ids))``.
        for data in datas:
            flat = np.empty(data.shape, data.dtype, order="C")
            flat[unique_ids] = data
            yield cls(flat, bounds)

    # For pickle.
    def __getstate__(self):
        # Unfortunately this will lose the memory efficiency of letting starts
        # and ends overlap.
        # I'm choosing to version this pickle function so that, if I fix the
        # above, then I can avoid version mismatch chaos.
        from rockhopper import __version__
        return 0, __version__, self.flat, self.starts, self.ends

    def __setstate__(self, state):
        pickle_version, rockhopper_version, *state = state
        if pickle_version > 0:
            import pickle
            raise pickle.UnpicklingError(
                "This ragged array was pickled with a newer version of "
                f"rockhopper ({rockhopper_version}) which wrote its pickles "
                f'differently. Running:\n    pip install '
                f'"rockhopper >= {rockhopper_version}"\nshould fix this.')
        self.__init__(*state)


ragged_array = RaggedArray.from_nested


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


def sub_enumerate(ids, id_max):
    """Wrapper of :c:`sub_enumerate()` from src/ragged_array.c

    Args:
        ids (numpy.ndarray):
            A group number for each element.
        id_max (int):
            A strict upper bound for the **ids**.

    Returns:
        counts (numpy.ndarray):
            :py:`counts[x] := ids.count(x)`.
        sub_ids (numpy.ndarray):
            :py:`sub_ids[i] := ids[:i].count(ids[i])`.

    Raises:
        IndexError:
            If either :py:`(0 <= ids).all() or :py`(ids < id_max).all()`
            are not satisfied.

    """
    ids = np.ascontiguousarray(ids, dtype=np.intc)
    counts = np.zeros(int(id_max), np.intc)
    sub_ids = np.empty_like(ids)
    slug.dll.sub_enumerate(ptr(ids), ids.size, ptr(counts), ptr(sub_ids))
    return counts, sub_ids
