# -*- coding: utf-8 -*-
"""
"""

import sys
import ctypes

import pytest

from rockhopper._ragged_array import slug

pytestmark = pytest.mark.order(1)


def log_range(start, stop, base):
    while start < stop:
        # These sequences give a quick way to test the full range of an
        # integer type.
        yield start
        start *= base


def test_log_range():
    assert list(log_range(1, 10, 3)) == [1, 3, 9]


@pytest.mark.parametrize("int_base", range(4))
def test_endian_swap(int_base):
    """Test the family of :c:`swap_endian_xx()` functions."""
    bytes = (1 << int_base)
    bits = bytes * 8

    swap = getattr(slug.dll, f"swap_endian_{bits}")

    for i in log_range(1, 1 << bytes, 3):
        assert swap(i).to_bytes(bytes, "big") == i.to_bytes(bytes, "little")


def test_is_big_endian():
    """Test :c:`is_big_endian()` matched :attr:`sys.byteorder`."""
    assert slug.dll.is_big_endian() == (sys.byteorder == "big")


def f_ptr(f):
    """Get the raw memory address of a :mod:`ctypes` function pointer."""
    return ctypes.cast(f, ctypes.c_void_p).value


@pytest.mark.parametrize("int_base", range(4))
@pytest.mark.parametrize("byteorder", ["little", "big"])
def test_int_write(int_base, byteorder):
    """
    Test the family of :c:`write_xx()` and :c:`write_swap_xx()` integer
    writing functions and the selector :c:`choose_int_write()`.
    """
    bytes = 1 << int_base
    bits = 8 * bytes
    native = sys.byteorder == byteorder

    # The real return type of `choose_int_write()` is `IntWrite` which is a
    # typedef (which cslug doesn't support) to a function pointer (which
    # cslug also doesn't support). We only need to test which function it
    # returns so raw a void pointer is sufficient.
    slug.dll.choose_int_write.restype = ctypes.c_void_p

    # Get the writer we expect to get.
    name = f"write_{bits}" if native else f"write_swap_{bits}"
    write = getattr(slug.dll, name)

    # Check it matches the output of choose_int_write()`.
    assert slug.dll.choose_int_write(int_base,
                                     byteorder == "big") == f_ptr(write)

    # Try writing an integer with it.
    x = 0x1122334455667788 & ((1 << bits) - 1)
    out = ctypes.create_string_buffer(bytes)
    write(x, out)
    assert list(out[:]) == list(x.to_bytes(bytes, byteorder))

    read = getattr(slug.dll, name.replace("write", "read"))
    assert read(out) == x
