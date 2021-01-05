# -*- coding: utf-8 -*-
"""
"""

import numpy as np
import pytest

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
def test_dump(dtype, byteorder):
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
