#ifndef KHARON_ARENA_H
#define KHARON_ARENA_H

#include <stddef.h>

// A bump allocator over one slab. We own the memory and hand out aligned
// chunks; no per-op cudaMalloc. Use one arena per buffer class (params, grads,
// activations, optimizer state) so high-water is tracked per class.
typedef struct {
  char name[32];
  unsigned char *base;
  size_t cap, off, high;
  int on_device;
} Arena;

#ifdef __cplusplus
extern "C" {
#endif

Arena  arena_create(const char *name, size_t cap, int on_device);
void  *arena_alloc(Arena *a, size_t bytes);
void   arena_reset(Arena *a);
void   arena_destroy(Arena *a);
size_t arena_used(const Arena *a);
size_t arena_high(const Arena *a);

#ifdef __cplusplus
}
#endif

#endif
