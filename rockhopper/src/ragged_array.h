// -*- coding: utf-8 -*-

#ifndef ragged_array_H
#define ragged_array_H


typedef struct RaggedArray
{
  void * flat;
  int itemsize;
  int length;
  int * starts;
  int * ends;
} RaggedArray;


#endif
