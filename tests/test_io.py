# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import pytest
from cslug import ptr

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

    bin = self.dumps(lengths_dtype=dtype)
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
                          lengths_dtype=dtype)

    parsed = RaggedArray.loads(bin, dtype=self.dtype, lengths_dtype=dtype)
    assert np.array_equal(self.starts, parsed.starts)
    assert np.array_equal(self.ends, parsed.ends)
    assert np.array_equal(self.flat, parsed.flat)


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

    parsed = RaggedArray.loads(self.dumps(), dtype=np.dtype(np.intc) * 3)
    assert np.array_equal(self.starts, parsed.starts)
    assert np.array_equal(self.ends, parsed.ends)
    assert np.array_equal(self.flat, parsed.flat)
