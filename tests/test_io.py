# -*- coding: utf-8 -*-
"""
"""

import pickle

import numpy as np
import pytest
from cslug import ptr
from hypothesis import given, strategies, settings, Verbosity, example

from rockhopper import RaggedArray
from rockhopper._ragged_array import _2_power, _big_endian

pytestmark = pytest.mark.order(3)


def test_2_power():
    assert _2_power(np.int8) == 0
    assert _2_power(np.int16) == 1
    assert _2_power(np.int32) == 2
    assert _2_power(np.int64) == 3
    assert _2_power(np.uint64) == 3

    with pytest.raises(TypeError):
        _2_power(1)

    for dtype in [np.uint8, np.int8, np.intc, np.int16, np.uint32, np.int64]:
        assert 1 << _2_power(dtype) == dtype().itemsize


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32])
@pytest.mark.parametrize("byteorder", "<>=|")
def test_dump_load(dtype, byteorder):
    dtype = np.dtype(dtype).newbyteorder(byteorder)
    flat = np.arange(5, dtype=np.int8)
    self = RaggedArray.from_lengths(flat, [2, 3, 0])

    _byteorder = "big" if _big_endian(dtype) else "little"
    _bin_int = lambda x: int.to_bytes(x, dtype.itemsize, _byteorder)

    bin = self.dumps(ldtype=dtype)
    target = (_bin_int(2), flat[0:2].tobytes(),
              _bin_int(3), flat[2:5].tobytes(),
              _bin_int(0), b"")  # yapf: disable

    # Convert to lists only to make the pytest traceback more readable.
    assert list(bin) == list(b"".join(target))

    from rockhopper._ragged_array import slug
    assert slug.dll.count_rows(ptr(bin), len(bin), _2_power(dtype),
                               _big_endian(dtype), flat.itemsize) == len(self)

    with pytest.raises(ValueError):
        RaggedArray.loads(bin.tobytes() + b"\x01", dtype=self.dtype,
                          ldtype=dtype)

    parsed, consumed = RaggedArray.loads(bin, dtype=self.dtype, ldtype=dtype)
    assert np.array_equal(self.starts, parsed.starts)
    assert np.array_equal(self.ends, parsed.ends)
    assert np.array_equal(self.flat, parsed.flat)
    assert consumed == len(bin)


int_types = [
    np.uint8, np.uint16, np.uint32, np.uint64,
    np.int8, np.int16, np.int32, np.int64,
]
blob = bytes(range(256)) + bytes(range(256))[::-1]


@pytest.mark.parametrize("dtype", int_types)
@pytest.mark.parametrize("ldtype", int_types)
def test_loads_pointer_overflow_guard(dtype, ldtype):
    """Test that the check for pointer overflowing caused by reading a huge row
    length works."""
    for i in range(-30, len(blob)):
        try:
            RaggedArray.loads(blob[i: i+30], dtype=dtype, ldtype=ldtype)
        except ValueError:
            pass


@pytest.mark.parametrize("dtype", int_types)
@pytest.mark.parametrize("ldtype", int_types)
def test_fuzz_loads(dtype, ldtype):
    """Scan for possible segfaults.

    All invalid inputs must lead to a ValueError rather than a seg-fault or
    RaggedArray.loads() could be tricked into reading arbitrary memory addresses
    by a maliciously constructed invalid data file.

    """
    @given(strategies.binary())
    @example(b'\xc0\\\\\xb0\x93\x91\xff\xffpEfe\x167\xee')
    def fuzz(x):
        print(x)
        try:
            self, _ = RaggedArray.loads(x, dtype=dtype, ldtype=ldtype)
        except ValueError:
            pass
        else:
            assert self.dumps(ldtype=ldtype).tobytes() == x

    fuzz()


def test_dump_byteorder():
    self = RaggedArray.from_nested([[0x0109, 0x0208, 0x0307]], dtype=np.uint16)

    bin = list(self.astype(self.dtype.newbyteorder(">")).dumps(ldtype=np.uint8))
    assert bin == [3, 1, 9, 2, 8, 3, 7]

    bin = list(self.astype(self.dtype.newbyteorder("<")).dumps(ldtype=np.uint8))
    assert bin == [3, 9, 1, 8, 2, 7, 3]


def test_3d():
    self = RaggedArray.from_nested([
        [[0, 1, 2], [3, 4, 5]],
        [[6, 7, 8], [9, 10, 11]],
        [[12, 13, 14], [15, 16, 17], [18, 19, 20]],
        [],
    ], dtype=np.intc)

    # By using the same dtype as starts and ends (intc), it is safe (and far
    # easier to read) to think of the raw binary from ``self.dumps()`` as a
    # series of integers.

    target = [2, 0, 1, 2, 3, 4, 5,
              2, 6, 7, 8, 9, 10, 11,
              3, 12, 13, 14, 15, 16, 17, 18, 19, 20,
              0]  # yapf: disable

    assert np.frombuffer(self.dumps(), np.intc).tolist() == target

    parsed, _ = RaggedArray.loads(self.dumps(), dtype=np.dtype(np.intc) * 3)
    assert np.array_equal(self.starts, parsed.starts)
    assert np.array_equal(self.ends, parsed.ends)
    assert np.array_equal(self.flat, parsed.flat)


@pytest.mark.parametrize("ldtype", [np.uint8, np.uint16, np.uint32])
def test_empty(ldtype):
    self, consumed = RaggedArray.loads(b"", None, ldtype=ldtype)
    assert len(self) == 0
    assert len(self.flat) == 0
    assert consumed == 0


def test_corruption():
    """Invalid input should raise a deliberate :class:`ValueError`. Not a
    seg-fault."""

    bin = np.array([2, 100, 101, 1, 102, 0], np.uint16).tobytes()

    # End halfway through the 1st length.
    with pytest.raises(ValueError, match="through a row"):
        RaggedArray.loads(bin[:1], np.uint16, ldtype=np.uint16)
    with pytest.raises(ValueError, match="leaves -1 bytes for the flat data"):
        RaggedArray.loads(bin[:1], np.uint16, ldtype=np.uint16, rows=1)

    assert len(RaggedArray.loads(bin[:1], None, rows=0)[0]) == 0

    # End after the 1st row length but before the row data.
    with pytest.raises(ValueError, match="through a row"):
        RaggedArray.loads(bin[:2], np.uint16, ldtype=np.uint16)

    # Again but with rows specified.
    with pytest.raises(ValueError, match="Only 0 out of .* 1 rows were read."):
        RaggedArray.loads(bin[:2], np.uint16, ldtype=np.uint16, rows=1)

    # A full row of binary data - should work.
    RaggedArray.loads(bin[:6], ldtype=np.uint16, dtype=np.uint16)

    # But not of the user expects more rows.
    with pytest.raises(ValueError, match="Only 1 out of .* 2 rows were read."):
        RaggedArray.loads(bin[:6], ldtype=np.uint16, dtype=np.uint16, rows=2)

    # Be sure the empty last row doesn't get lost.
    ragged, consumed = RaggedArray.loads(bin, ldtype=np.uint16, dtype=np.uint16)
    assert len(ragged) == 3
    assert consumed == len(bin)
    RaggedArray.loads(bin, ldtype=np.uint16, dtype=np.uint16, rows=3)


def test_overflow():
    """Test dumps() for large row lengths with too small row-length dtype."""

    self = RaggedArray.from_lengths(np.arange(1000), [0, 150, 255, 256, 300])
    with pytest.raises(OverflowError,
                       match="Row 3 with length 256 is .* an uint8 integer."):
        self.dumps(ldtype=np.uint8)

    self.dumps(np.int16)


class ImplementationFromTheFuture(RaggedArray):
    """A subclass of RaggedArray which, when pickling, pretends to use an
     pickling format from the future."""

    def __getstate__(self):
        return 1000, "12.34.56", ("some", "nonsense")


def test_pickle():
    self = RaggedArray.from_nested([
        ["cake", "biscuits"],
        ["socks"],
        ["orange", "lemon", "pineapple"],
    ])

    copied = pickle.loads(pickle.dumps(self))
    assert np.array_equal(self.starts, copied.starts)
    assert np.array_equal(self.ends, copied.ends)
    assert np.array_equal(self.flat, copied.flat)


def test_pickle_versioning():
    self = ImplementationFromTheFuture([], [])
    pickled = pickle.dumps(self)
    with pytest.raises(
            pickle.UnpicklingError,
            match=r'(?s).* version of rockhopper \(12.34.56\) which '
            r'.* install "rockhopper >= 12.34.56"\nshould'):
        pickle.loads(pickled)
