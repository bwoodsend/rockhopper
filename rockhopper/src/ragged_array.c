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


void dump(RaggedArray * self, void * out, int length_power, int big_endian) {
  IntWrite write = choose_int_write(length_power, big_endian);

  for (int i = 0; i < self -> length; i++) {

    int length = self -> ends[i] - self -> starts[i];

    write(length, out);
    out += (1 << length_power);

    memcpy(out, self -> flat + self -> starts[i] * self -> itemsize,
           length * self -> itemsize);
    out += length * self -> itemsize;

  }
}


int count_rows(void * raw, int raw_length, int length_power, int big_endian,
               int itemsize) {
  /* Pre-parse data fed to ``RaggedArray.loads()``.

  Returns the number of rows or -1 for an error.
  */

  IntRead read = choose_int_read(length_power, big_endian);

  int rows = 0;

  void * end = raw + raw_length;
  while (raw < end) {
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


void load(RaggedArray * self, void * raw, int raw_length, int length_power,
          int big_endian) {
  /* The workhorse behind ``RaggedArray.loads()``. */

  IntRead read = choose_int_read(length_power, big_endian);

  int start = self -> starts[0] = 0;
  void * raw_end = raw + raw_length;

  // Parse the array a row at a time:
  for (int row = 0; raw < raw_end; row++) {
    // Read the `length` of the row then move the `raw` pointer onto the row's
    // data.
    uint64_t length = read(raw);
    raw += (1 << length_power);

    // Copy the row data itself and move the input pointer to the next row.
    memcpy(self -> flat + start * self -> itemsize, raw,
           self -> itemsize * length);
    raw += length * self -> itemsize;

    // Set mark the end of this row. This is also writing the start of the next
    // row because `self -> ends` is the same arrays as`self -> starts` but
    // shifted along one item.
    start += length;
    self -> ends[row] = start;

  }
}
