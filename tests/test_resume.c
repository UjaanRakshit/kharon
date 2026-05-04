#include "model.h"
#include "refio.h"
#include "ckpt.h"
#include "rng.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define N 8

static int *idx, *tgt;
static long R;
static float LR, B1, B2, EPS, WD;

static float train_step(Model *m, Rng *rng) {
  rng_batch(rng, idx, tgt, R, m->cfg.vocab);
  model_set_input(m, idx, tgt);
  float loss = model_forward(m);
  model_backward(m);
  model_adamw_step(m, LR, B1, B2, EPS, WD);
  return loss;
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "tests/m1_ref.bin";
  RefFile *r = ref_load(path);
  Config cfg = {r->n_layer, r->d_model, r->n_head, r->vocab, r->seq, r->batch};
  R = (long)cfg.batch * cfg.seq;
  LR = r->lr; B1 = r->beta1; B2 = r->beta2; EPS = r->eps; WD = r->wd;
  idx = (int *)malloc(R * 4);
  tgt = (int *)malloc(R * 4);
  const uint64_t SEED = 1234;

  float full[2 * N], part[2 * N];

  // uninterrupted reference run
  Model *m = model_create(cfg);
  model_load_ref(m, r);
  Rng rng; rng_seed(&rng, SEED);
  for (int i = 0; i < 2 * N; i++) full[i] = train_step(m, &rng);
  model_free(m);

  // interrupted: N steps, checkpoint, reload into fresh model, N more
  m = model_create(cfg);
  model_load_ref(m, r);
  rng_seed(&rng, SEED);
  for (int i = 0; i < N; i++) part[i] = train_step(m, &rng);
  ckpt_save(m, rng.s, "build/m1.ckpt");
  model_free(m);

  m = model_create(cfg);
  Rng rng2;
  ckpt_load(m, &rng2.s, "build/m1.ckpt");
  for (int i = N; i < 2 * N; i++) part[i] = train_step(m, &rng2);
  model_free(m);

  int ok = 1;
  printf("resume bit-exact check (%d steps, checkpoint at %d):\n", 2 * N, N);
  for (int i = 0; i < 2 * N; i++) {
    int same = (full[i] == part[i]);
    if (i == N - 1 || i == N || i == 2 * N - 1)
      printf("  step %2d  full=%.8f  part=%.8f  %s\n", i, full[i], part[i], same ? "" : "<< DIFF");
    ok &= same;
  }
  free(idx); free(tgt);
  ref_free(r);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
