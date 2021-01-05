// -*- coding: utf-8 -*-

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#include "endians.h"


bool is_big_endian() {
  /* return 1 for big endian, 0 for little endian. */

  volatile uint32_t i=0x66000055;
  uint8_t first_byte = *((uint8_t*)(&i));

  if (first_byte == 0x66)
    return 1;
  // first_byte == 0x55
  return 0;
}

/* --- Endian swapping functions --- */

// These are taken from: https://stackoverflow.com/a/2637138

uint8_t swap_endian_8(uint8_t x) {
  return x;
}

uint16_t swap_endian_16(uint16_t x) {
  return (x >> 8) | (x << 8);
}

uint32_t swap_endian_32(uint32_t x) {
  x = ((x << 8) & 0xFF00FF00 ) | ((x >> 8) & 0xFF00FF );
  return (x << 16) | (x >> 16);
}

uint64_t swap_endian_64(uint64_t x) {
  x = ((x <<  8) & 0xFF00FF00FF00FF00ULL ) | ((x >>  8) & 0x00FF00FF00FF00FFULL);
  x = ((x << 16) & 0xFFFF0000FFFF0000ULL ) | ((x >> 16) & 0x0000FFFF0000FFFFULL);
  return (x << 32) | (x >> 32);
}


/* --- Write integers with arbitrary sizes and byte-orders --- */

void write_8 (uint64_t x, void * out) { uint8_t  y = x; memcpy(out, &y, 1); }
void write_16(uint64_t x, void * out) { uint16_t y = x; memcpy(out, &y, 2); }
void write_32(uint64_t x, void * out) { uint32_t y = x; memcpy(out, &y, 4); }
void write_64(uint64_t x, void * out) { uint64_t y = x; memcpy(out, &y, 8); }

void write_swap_8 (uint64_t x, void * out) { uint8_t  y = x; y = swap_endian_8 (y); memcpy(out, &y, 1); }
void write_swap_16(uint64_t x, void * out) { uint16_t y = x; y = swap_endian_16(y); memcpy(out, &y, 2); }
void write_swap_32(uint64_t x, void * out) { uint32_t y = x; y = swap_endian_32(y); memcpy(out, &y, 4); }
void write_swap_64(uint64_t x, void * out) { uint64_t y = x; y = swap_endian_64(y); memcpy(out, &y, 8); }


IntWrite int_writers[8] = {
    write_8, write_16, write_32, write_64,
    write_swap_8, write_swap_16, write_swap_32, write_swap_64
};


IntWrite choose_int_write(int power, bool big_endian) {
  /* Choose a write function pointer for a given int type and endian. */
  return (IntWrite) _choose_int_read_write(
                      power, big_endian, (void *) int_writers);
}


/* --- Read integers with arbitrary sizes and byte-orders --- */

uint64_t read_8 (void * x) { return *((uint8_t *) x); }
uint64_t read_16(void * x) { return *((uint16_t *) x); }
uint64_t read_32(void * x) { return *((uint32_t *) x); }
uint64_t read_64(void * x) { return *((uint64_t *) x); }

uint64_t read_swap_8 (void * x) { return swap_endian_8 (read_8 (x)); }
uint64_t read_swap_16(void * x) { return swap_endian_16(read_16(x)); }
uint64_t read_swap_32(void * x) { return swap_endian_32(read_32(x)); }
uint64_t read_swap_64(void * x) { return swap_endian_64(read_64(x)); }


IntRead int_readers[8] = {
    read_8, read_16, read_32, read_64,
    read_swap_8, read_swap_16, read_swap_32, read_swap_64
};


IntRead choose_int_read(int power, bool big_endian) {
  /* Choose a read function pointer for a given int type and endian. */
  return (IntRead) _choose_int_read_write(
                      power, big_endian, (void *) int_readers);
}


/* --- */

void * _choose_int_read_write(int power, bool big_endian, void ** list) {
  /* Choose a read/write function pointer for a given int type and endian. */
  int swap_endian = (big_endian != is_big_endian());
  int ordinal = power + 4 * swap_endian;
  if (ordinal < 0 || ordinal > 8)
    return 0;
  return list[ordinal];
}
