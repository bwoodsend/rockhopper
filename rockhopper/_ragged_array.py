# -*- coding: utf-8 -*-
"""
"""

from typing import Union, Type, Tuple
import ctypes
import sys

import numpy as np
from cslug import CSlug, ptr, anchor, Header

from rockhopper import RequestMeError

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

dtype_like = Union[np.dtype, Type[np.generic]]


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

    def __init__(self, flat, starts, ends=None, dtype=None, check=True):
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
            check:
                If true (default), verify that **starts** and **ends** are
                valid (via :meth:`check`). Please only disable this if you need
                to a construct a ragged array by first creating an uninitialised
                array to then populating it. Invalid arrays can lead to
                seg-faults.

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
            raise RequestMeError(
                "Flat lengths >= 2^31  are disabled at compile time to save "
                "memory at runtime.")

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

        if check:
            self.check()

    def check(self):
        """Verify that this array has valid shape.

        Raises:
            ValueError:
                If :attr:`starts` and :attr:`ends` are not of the same length.
            ValueError:
                If any row has a negative length. (0 length rows are ok.)
            IndexError:
                If any row starts (:attr:`starts`) are negative.
            IndexError:
                If any row ends (:attr:`ends`) are out of bounds (>= len(flat)).

        """
        if len(self.starts) != len(self.ends):
            raise ValueError(f"The lengths of starts ({len(self.starts)}) and "
                             f"ends ({len(self.ends)}) do not match.")

        for index in _violates(self.starts > self.ends):
            raise ValueError(f"Row {index}, "
                             f"starting at flat[{self.starts[index]}] "
                             f"and ending at flat[{self.ends[index]}], "
                             f"has a negative length "
                             f"({self.ends[index] - self.starts[index]}).")

        for index in _violates(self.starts < 0):
            raise IndexError(f"Invalid value in `starts` attribute: "
                             f"starts[{index}] = {self.starts[index]} < 0")

        for index in _violates(self.ends > len(self.flat)):
            raise IndexError(f"Invalid value in `ends` attribute: "
                             f"ends[{index}] = {self.ends[index]} >= "
                             f"len(flat) = {len(self.flat)}")

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

    def byteswap(self, inplace=False):
        """Swap endian. Analogous to :meth:`numpy.ndarray.byteswap`.

        Args:
            inplace:
                If true, modify this array. Otherwise create a new one.
        Returns:
            Either this array or a new ragged array with opposite byte order.

        The byteorder of the :attr:`starts` and :attr:`ends` arrays are not
        touched.

        """
        if inplace:
            self.flat.byteswap(inplace=inplace)
            return self
        return type(self)(self.flat.byteswap(), self.starts, self.ends,
                          self.dtype)

    @classmethod
    def from_lengths(cls, flat, lengths, dtype=None):
        bounds = np.empty(len(lengths) + 1, dtype=np.intc)
        bounds[0] = 0
        np.cumsum(lengths, out=bounds[1:])
        return cls(flat, bounds, dtype=dtype)

    @classmethod
    def from_nested(cls, nested, dtype=None):
        _nested = [np.asarray(i, dtype=dtype) for i in nested if len(i)]
        if _nested:
            flat = np.concatenate(_nested)
        else:
            flat = np.empty(0, dtype=dtype)
        lengths = [len(i) for i in nested]
        return cls.from_lengths(flat, lengths, dtype=dtype)

    def __getitem__(self, item) -> Union['RaggedArray', np.ndarray]:
        index = self.__index_item__(item)
        if isinstance(index, type(self)):
            return index
        return self.flat[index]

    def __setitem__(self, key, value):
        index = self.__index_item__(key)
        if isinstance(index, type(self)):
            raise RequestMeError
        self.flat[index] = value

    def __index_item__(self, item):
        """The brain behind __getitem__() and __setitem__().

        To avoid having to write everything twice (for set and get item), this
        function returns :attr:`flat` indices which may then be used as
        ``return flat[indices]`` or ``flat[indices] = value``.

        Unfortunately, there are a lot of permutations of possible input types.
        Some of these permutations return another RaggedArray which should be
        returned directly by getitem and (once I've implemented vectorisation)
        wrote to directly by setitem.

        """
        # 2D indexing i.e. ragged[rows, columns]
        if isinstance(item, tuple) and len(item) == 2:
            rows, columns = item
            if isinstance(columns, slice):
                if _null_slice(columns):
                    # Case self[rows, :] should be simplified to ragged[rows]
                    return self.__index_item__(rows)
                # Case self[rows, slice]
                return self.__index_item_number_slice__(rows, columns)
            # Case self[rows, numerical column numbers]
            return self.__index_item_any_number__(rows, columns)

        # 3+D indexing.
        if isinstance(item, tuple) and len(item) > 2:
            if self.itemshape:
                # Covert to self[2D index, *other indices].
                indices = self.__index_item__(item[:2])
                if isinstance(indices, type(self)):
                    raise RequestMeError("Returning ragged arrays from >2D "
                                         "indices is not implemented.")
                return (indices, *item[2:])
            raise IndexError(
                f"Too many indices for ragged array: maximum allowed is 2 but "
                f"{len(item)} were given.")

        # 1D indexing (ragged[rows]).
        if np.isscalar(item):
            # A single row number implies just a regular array output.
            return slice(self.starts[item], self.ends[item])
        # Whereas any of slicing, bool masks, arrays of row numbers, ...
        # return another ragged array.
        return type(self)(self.flat, self.starts[item], self.ends[item])

    def __index_item_any_number__(self, rows, columns):
        """Indices for self[rows, columns] where **columns** is numeric (not a
        slice or bool mask)."""
        rows_is_array_like = not (isinstance(rows, slice) or rows is None)
        if rows_is_array_like:
            rows = np.asarray(rows)
            assert rows.dtype != object
        columns = np.asarray(columns)
        assert columns.dtype != object

        starts = self.starts[rows]
        ends = self.ends[rows]
        if not rows_is_array_like:
            columns = columns[np.newaxis]
        while starts.ndim < columns.ndim:
            starts = starts[..., np.newaxis]
            ends = ends[..., np.newaxis]

        lengths = ends - starts
        out_of_bounds = (columns < -lengths) | (columns >= lengths)
        for index in _violates(out_of_bounds):
            rows = np.arange(len(self))[rows]
            while rows.ndim < columns.ndim:
                rows = rows[..., np.newaxis]

            rows, columns, lengths = np.broadcast_arrays(rows, columns, lengths)
            raise IndexError(f"Index {columns[index]} is out of bounds for row "
                             f"{rows[index]} with size {lengths[index]}")
        columns = np.where(columns < 0, columns + lengths, columns)
        return starts + columns

    def __index_item_number_slice__(self, rows, columns: slice):
        """Indices for self[rows, columns] where **columns** is a slice."""
        if columns.step not in (1, None):
            raise RequestMeError(
                "A stepped columns index ragged[x, ::step] is not implemented "
                "as it would require strided ragged arrays (which are also not "
                "implemented).")

        starts = self.starts[rows]
        ends = self.ends[rows]
        lengths = ends - starts

        if columns.start is None:
            new_starts = starts
        else:
            new_starts = starts + _wrap_negative(columns.start, lengths)
            new_starts.clip(starts, ends, out=new_starts)

        if columns.stop is None:
            new_ends = ends
        else:
            new_ends = starts + _wrap_negative(columns.stop, lengths)
            new_ends.clip(starts, ends, out=new_ends)
            new_ends.clip(new_starts, out=new_ends)

        return type(self)(self.flat, *np.broadcast_arrays(new_starts, new_ends))

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
        new = type(self)(flat, bounds[:-1], bounds[1:], self.dtype, check=False)
        slug.dll.repack(self._c_struct._ptr, new._c_struct._ptr)
        return new

    def dumps(self, ldtype=np.intc):
        """Serialise into a :class:`memoryview`.

        Args:
            ldtype (Union[numpy.dtype, Type[numpy.generic]]):
                Integer type for the row lengths.
        Returns:
            memoryview:
                A bytes-like binary blob.

        The binary format is an undelimited sequence of ``(len(row), row)``
        pairs. A pure Python approximation would be::

            b"".join((len(row).tobytes() + row.tobytes() for row in ragged_array))

        The integer types of the row lengths can be controlled by the
        **ldtype** parameter. To change the type or byteorder of the data
        itself, cast to that type with :meth:`astype` then call this function.

        """
        ldtype = np.dtype(ldtype)

        # --- Work out how many bytes the output will need. ---

        # The total length of the flat data. Note, `self.flat.size` would not be
        # a safe shortcut unless `self.repacked()` has been called 1st.
        length = (self.ends - self.starts).sum() * self.itemsize
        # And the lengths of the lengths...
        length += len(self) * ldtype.itemsize

        # Allocate `length` bytes to write to. `numpy.empty()` seems to be one
        # of the only ways to create a lump of memory in Python without wasting
        # time initialising it.
        out = np.empty(length, dtype=np.byte)

        failed_row = slug.dll.dump(self._c_struct._ptr, ptr(out),
                                   _2_power(ldtype), _big_endian(ldtype))
        if failed_row != -1:
            raise OverflowError(
                f"Row {failed_row} with length {len(self[failed_row])} "
                f"is too long to write with an {ldtype.name} integer.")
        return out.data

    @classmethod
    def loads(cls, bin, dtype, rows=-1,
              ldtype=np.intc) -> Tuple['RaggedArray', int]:
        """Deserialize a ragged array. This is the reciprocal of :meth:`dumps`.

        Args:
            bin (bytes):
                Raw data to unpack.
            dtype (Union[numpy.dtype, Type[numpy.generic]]):
                Data type of the row contents in **bin**.
            rows (int):
                Number of rows to parse. Defaults to :py:`-1` for unknown.
            ldtype (Union[numpy.dtype, Type[numpy.generic]]):
                Integer type of the row lengths in **bin**.
        Returns:
            RaggedArray:
                The deserialised ragged array.
            int:
                The number of bytes from **bin** consumed.
        Raises:
            ValueError:
                If **bin** ends prematurely or in the middle of a row. This is
                indicative of either data corruption or, more likely, muddling
                of dtypes.

        """
        dtype = np.dtype(dtype)
        ldtype = np.dtype(ldtype)

        # We need to know how many rows there will be in this new ragged array
        # before creating and populating it.
        if rows == -1:
            # If it's not already known then it has to be counted.
            rows = slug.dll.count_rows(ptr(bin), len(bin), _2_power(ldtype),
                                       _big_endian(ldtype), dtype.itemsize)
            if rows == -1:
                # `count_rows()` returns -1 on error.
                raise ValueError(
                    "Raw `bin` data ended mid way through a row. Either this "
                    "data is corrupt or the dtype(s) given are incorrect.")

            # Run again with known number of `rows`.
            return cls.loads(bin, dtype, rows, ldtype)

        free = len(bin) - rows * ldtype.itemsize
        items = free // dtype.itemsize
        if items < 0:
            raise ValueError(
                f"With `bin` of length {len(bin)}, {rows} rows of "
                f"{ldtype.itemsize} byte lengths leaves {free} bytes "
                f"for the flat data. Perhaps your data types are wrong?")

        self = cls(np.empty(items, dtype=dtype), np.empty(rows + 1, np.intc),
                   check=False)

        bin_consumed = ctypes.c_size_t(0)

        _rows = slug.dll.load(self._c_struct._ptr, ptr(bin), len(bin),
                              ctypes.byref(bin_consumed), rows,
                              _2_power(ldtype), _big_endian(ldtype))
        if _rows < rows:
            raise ValueError(
                f"Raw `bin` data ended too soon. "
                f"Only {_rows} out of the requested {rows} rows were read. "
                f"Either this data is corrupt or the dtype(s) given are "
                "incorrect.")

        return self, bin_consumed.value

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

    def tolist(self):
        """Convert to a list of lists. This is analogous to
        :meth:`numpy.ndarray.tolist` and is the reciprocal of
        :meth:`from_nested`."""
        return sum(map(np.ndarray.tolist, self.to_rectangular_arrays()), [])

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


def _violates(mask: np.ndarray):
    """Yield the index of the first true element, if any, of the boolean array
    **mask**. Otherwise don't yield at all."""
    if np.any(mask):
        index = np.argmax(mask)
        yield np.unravel_index(index, mask.shape) if mask.ndim != 1 else index


def _null_slice(s: slice):
    """Return true if a slice does nothing e.g. list[:]"""
    return s.start is s.step is s.stop is None


def _wrap_negative(indices, lengths):
    """Add **lengths** to **indices** which are negative. Mimics Python's usual
     list[-1] => list[len(list) - 1] behaviour."""
    return np.where(indices < 0, indices + lengths, indices)
