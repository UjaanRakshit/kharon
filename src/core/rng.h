#ifndef KHARON_RNG_H
#define KHARON_RNG_H

#include <stdint.h>

// xorshift64*: tiny, deterministic, state is one u64 so it checkpoints trivially.
typedef struct { uint64_t s; } Rng;

static inline void rng_seed(Rng *r, uint64_t seed) { r->s = seed ? seed : 0x9E3779B97F4A7C15ULL; }

static inline uint32_t rng_u32(Rng *r) {
  uint64_t x = r->s;
  x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
  r->s = x;
  return (uint32_t)((x * 0x2545F4914F6CDD1DULL) >> 32);
}

// Synthetic next-token batch: fixed corpus stand-in for M1's determinism tests.
static inline void rng_batch(Rng *r, int *idx, int *tgt, long n, int vocab) {
  for (long i = 0; i < n; i++) idx[i] = (int)(rng_u32(r) % (uint32_t)vocab);
  for (long i = 0; i < n; i++) tgt[i] = (int)(rng_u32(r) % (uint32_t)vocab);
}

#endif
