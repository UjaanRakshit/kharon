#include "arena.h"
#include "kharon.h"
#include <string.h>
#include <cuda_runtime.h>

#define ALIGN 256

Arena arena_create(const char *name, size_t cap, int on_device) {
  Arena a;
  memset(&a, 0, sizeof(a));
  strncpy(a.name, name, sizeof(a.name) - 1);
  a.cap = cap;
  a.on_device = on_device;
  if (cap) {
    if (on_device) CK(cudaMalloc((void **)&a.base, cap));
    else           a.base = (unsigned char *)malloc(cap);
    if (!a.base) DIE("arena '%s': alloc of %zu bytes failed", name, cap);
  }
  return a;
}

void *arena_alloc(Arena *a, size_t bytes) {
  size_t off = (a->off + (ALIGN - 1)) & ~(size_t)(ALIGN - 1);
  if (off + bytes > a->cap)
    DIE("arena '%s' OOM: want %zu at %zu of %zu", a->name, bytes, off, a->cap);
  a->off = off + bytes;
  if (a->off > a->high) a->high = a->off;
  return a->base + off;
}

void arena_reset(Arena *a) { a->off = 0; }

void arena_destroy(Arena *a) {
  if (!a->base) return;
  if (a->on_device) cudaFree(a->base);
  else              free(a->base);
  a->base = NULL;
  a->cap = a->off = a->high = 0;
}

size_t arena_used(const Arena *a) { return a->off; }
size_t arena_high(const Arena *a) { return a->high; }
