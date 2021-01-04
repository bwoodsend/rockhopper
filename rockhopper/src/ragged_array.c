// -*- coding: utf-8 -*-

#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include "ragged_array.h"


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
