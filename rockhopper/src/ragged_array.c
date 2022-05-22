// -*- coding: utf-8 -*-

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "ragged_array.h"
#include "endians.h"


void repack(RaggedArray * old, RaggedArray * new) {
  int new_start = new -> starts[0] = 0;
  for (int i = 0; i < old -> length; i++) {
    int start = old -> starts[i];
    int end = old -> ends[i];
    int size = end - start;
    memcpy(new -> flat + new_start * old -> itemsize,
           old -> flat + start * old -> itemsize,
           size * old -> itemsize);
    new_start += size;
    new -> ends[i] = new_start;
  }
}


int dump(RaggedArray * self, void * out, int length_power, int big_endian) {
  IntWrite write = choose_int_write(length_power, big_endian);
  int max_length = 1 << (8 << length_power);

  for (int i = 0; i < self -> length; i++) {

    int length = self -> ends[i] - self -> starts[i];

    // Escape if given a value too large to fit in the chosen integer type.
    if ((1 << length_power < (int) sizeof(int)) && (length >= max_length)) {
      return i;
     }

    write(length, out);
    out += (1 << length_power);

    memcpy(out, self -> flat + self -> starts[i] * self -> itemsize,
           length * self -> itemsize);
    out += length * self -> itemsize;

  }
  // Return -1 to indicate that no row lengths were too long.
  return -1;
}


int count_rows(void * raw, int raw_length, int length_power, int big_endian,
               int itemsize) {
  /* Pre-parse data fed to ``RaggedArray.loads()``.

  Returns the number of rows or -1 for an error.
  */

  IntRead read = choose_int_read(length_power, big_endian);

  int rows = 0;

  void * start = raw;
  void * end = raw + raw_length;
  while (raw <= end - (1 << length_power) && raw >= start) {
    uint64_t length = read(raw);
    raw += (1 << length_power);
    raw += length * itemsize;
    rows ++;
  }

  if (raw == end)
    return rows;

  // This data is corrupt.
  return -1;
}


uint64_t load(RaggedArray * self, void * raw, uint64_t raw_length,
              size_t * raw_consumed, int rows, int length_power,
              int big_endian) {
  /* The workhorse behind ``RaggedArray.loads()``. */

  IntRead read = choose_int_read(length_power, big_endian);

  int start = self -> starts[0] = 0;
  void * raw_start = raw;
  void * raw_end = raw + raw_length;

  // Parse the array a row at a time:
  for (int row = 0; row < rows; row++) {

    // Escape if there is not enough input left to read another length. This
    // shall be propagated as an error in Python.
    if (raw > raw_end - (1 << length_power)) return row;

    // Read the `length` of the row then move the `raw` pointer onto the row's
    // data.
    uint64_t length = read(raw);
    raw += (1 << length_power);

    // Again, escape if there is not enough remaining input to contain the
    // expected row.
    if (raw > raw_end - length * self -> itemsize) return row;

    // Copy the row data itself and move the input pointer to the next row.
    memcpy(self -> flat + start * self -> itemsize, raw,
           self -> itemsize * length);
    raw += length * self -> itemsize;

    // Mark the end of this row. This also sets the start of the next
    // row because `self -> ends` is the same array as`self -> starts` but
    // shifted along one item.
    start += length;
    self -> ends[row] = start;

  }
  // Record how many bytes of input have been used.
  *raw_consumed = raw - raw_start;

  return rows;
}


void sub_enumerate(int * ids, int len_ids, int * counts, int * enums) {
  /* Sub-enumerate grouped ids.

  The sub-enumerates **enums** are defined as:

    enums[i] := count(ids[:i] == ids[i])

  or how many times have we already seen ``ids[i]`` in ``ids``.

  A happy side-effect of this process is that it also counts how many of each
  value is in **ids**. These counts populate the **counts** array.
  */

  // For each id in ``ids``:
  for (int i = 0; i < len_ids; i++) {
    int id = ids[i];
    // Mark how many times we've already seen ``id``.
    enums[i] = counts[id];
    // Increment the count for ``id`` for future iterations.
    counts[id] += 1;
  }
}
