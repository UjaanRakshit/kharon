#include "data.h"
#include "kharon.h"
#include <stdlib.h>
#include <stdio.h>

void data_open(DataLoader *d, const char *path, int B, int T, uint64_t seed) {
  FILE *f = fopen(path, "rb");
  if (!f) DIE("data_open: cannot open %s", path);
  fseek(f, 0, SEEK_END);
  d->n = ftell(f);
  fseek(f, 0, SEEK_SET);
  if (d->n < (long)T + 1) DIE("data_open: corpus too small (%ld bytes)", d->n);
  d->buf = (unsigned char *)malloc(d->n);
  if (fread(d->buf, 1, d->n, f) != (size_t)d->n) DIE("data_open: short read");
  fclose(f);
  d->B = B; d->T = T;
  rng_seed(&d->rng, seed);
}

void data_next(DataLoader *d, int *idx, int *tgt) {
  for (int b = 0; b < d->B; b++) {
    long o = (long)(rng_u32(&d->rng) % (uint32_t)(d->n - d->T - 1));
    for (int t = 0; t < d->T; t++) {
      idx[b * d->T + t] = d->buf[o + t];
      tgt[b * d->T + t] = d->buf[o + t + 1];
    }
  }
}

void data_close(DataLoader *d) { free(d->buf); d->buf = NULL; }
