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
