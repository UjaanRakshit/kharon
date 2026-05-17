#ifndef KHARON_DATA_H
#define KHARON_DATA_H

#include "rng.h"

// Byte-level data loader: a raw byte file is the corpus (vocab=256). Each batch
// samples B random windows of length T+1; idx = window[:T], tgt = window[1:T+1].
typedef struct {
  unsigned char *buf;
  long n;
  int B, T;
  Rng rng;
} DataLoader;

#ifdef __cplusplus
extern "C" {
#endif

void data_open(DataLoader *d, const char *path, int B, int T, uint64_t seed);
void data_next(DataLoader *d, int *idx, int *tgt);   // host buffers [B*T]
void data_close(DataLoader *d);

#ifdef __cplusplus
}
#endif

#endif
