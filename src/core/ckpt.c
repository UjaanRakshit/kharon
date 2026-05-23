#include "ckpt.h"
#include "kharon.h"
#include <string.h>
#include <cuda_runtime.h>

#define CKPT_MAGIC "CKP1"

static void wr(FILE *f, const void *p, size_t n) {
  if (fwrite(p, 1, n, f) != n) DIE("ckpt: write failed");
}
static void rd(FILE *f, void *p, size_t n) {
  if (fread(p, 1, n, f) != n) DIE("ckpt: short read");
}

// Dump a flat device buffer through a host staging copy.
static void dump_dev(FILE *f, const void *dev, size_t bytes, void *stage) {
  CK(cudaMemcpy(stage, dev, bytes, cudaMemcpyDeviceToHost));
  wr(f, stage, bytes);
}
static void load_dev(FILE *f, void *dev, size_t bytes, void *stage) {
  rd(f, stage, bytes);
  CK(cudaMemcpy(dev, stage, bytes, cudaMemcpyHostToDevice));
}

void ckpt_save(const Model *m, uint64_t rng, const char *path) {
  FILE *f = fopen(path, "wb");
  if (!f) DIE("ckpt_save: cannot open %s", path);
  size_t pb = m->w_arena.off;
  size_t ob = m->dp_size > 1 ? (size_t)m->zero_n * 4 : pb;   // ZeRO-1: per-rank moment shard
  Config c = m->cfg;
  wr(f, CKPT_MAGIC, 4);
  wr(f, &c, sizeof(c));
  wr(f, &m->step, sizeof(int));
  wr(f, &rng, sizeof(uint64_t));
  wr(f, &pb, sizeof(size_t));
  wr(f, &ob, sizeof(size_t));
  void *stage = malloc(pb);
  dump_dev(f, m->w_arena.base, pb, stage);
  dump_dev(f, m->opt_m, ob, stage);
  dump_dev(f, m->opt_v, ob, stage);
  free(stage);
  fclose(f);
}

void ckpt_load(Model *m, uint64_t *rng, const char *path) {
  FILE *f = fopen(path, "rb");
  if (!f) DIE("ckpt_load: cannot open %s", path);
  char magic[4];
  Config c;
  size_t pb, ob;
  rd(f, magic, 4);
  if (memcmp(magic, CKPT_MAGIC, 4) != 0) DIE("ckpt_load: bad magic");
  rd(f, &c, sizeof(c));
  if (memcmp(&c, &m->cfg, sizeof(c)) != 0) DIE("ckpt_load: config mismatch");
  rd(f, &m->step, sizeof(int));
  rd(f, rng, sizeof(uint64_t));
  rd(f, &pb, sizeof(size_t));
  rd(f, &ob, sizeof(size_t));
  if (pb != m->w_arena.off) DIE("ckpt_load: param size mismatch");
  size_t want_ob = m->dp_size > 1 ? (size_t)m->zero_n * 4 : pb;
  if (ob != want_ob) DIE("ckpt_load: moment shard size mismatch");
  void *stage = malloc(pb);
  load_dev(f, m->w_arena.base, pb, stage);
  load_dev(f, m->opt_m, ob, stage);
  load_dev(f, m->opt_v, ob, stage);
  free(stage);
  fclose(f);
}
