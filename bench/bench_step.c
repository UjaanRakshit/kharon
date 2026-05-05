#include "model.h"
#include "refio.h"
#include "rng.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Times a full training step (forward + backward + AdamW) at the M1 dev config.
// This is the M1 baseline number to beat once kernels are optimized (M2+).
int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "tests/m1_ref.bin";
  int iters = argc > 2 ? atoi(argv[2]) : 200;
  RefFile *r = ref_load(path);
  Config cfg = {r->n_layer, r->d_model, r->n_head, r->vocab, r->seq, r->batch};
  long R = (long)cfg.batch * cfg.seq;

  Model *m = model_create(cfg);
  model_load_ref(m, r);
  int *idx = (int *)malloc(R * 4), *tgt = (int *)malloc(R * 4);
  Rng rng; rng_seed(&rng, 7);
  rng_batch(&rng, idx, tgt, R, cfg.vocab);
  model_set_input(m, idx, tgt);

  for (int i = 0; i < 20; i++) {  // warmup
    model_forward(m); model_backward(m);
    model_adamw_step(m, r->lr, r->beta1, r->beta2, r->eps, r->wd);
  }
  CK(cudaDeviceSynchronize());

  cudaEvent_t a, b;
  CK(cudaEventCreate(&a)); CK(cudaEventCreate(&b));
  CK(cudaEventRecord(a, 0));
  for (int i = 0; i < iters; i++) {
    model_forward(m); model_backward(m);
    model_adamw_step(m, r->lr, r->beta1, r->beta2, r->eps, r->wd);
  }
  CK(cudaEventRecord(b, 0));
  CK(cudaEventSynchronize(b));
  float ms = 0; CK(cudaEventElapsedTime(&ms, a, b));

  double per = ms / iters;
  double toks = R * 1000.0 / per;
  printf("config L=%d d=%d h=%d vocab=%d seq=%d batch=%d  (%d params)\n",
         cfg.n_layer, cfg.d_model, cfg.n_head, cfg.vocab, cfg.seq, cfg.batch, m->w.n_param);
  printf("fwd+bwd+step: %.3f ms/step over %d iters\n", per, iters);
  printf("throughput:   %.0f tokens/sec  (%ld tokens/step)\n", toks, R);
  printf("mem high-water: params=%.1fMB acts=%.1fMB bwd=%.1fMB adam=%.1fMB\n",
         m->w_arena.high / 1e6, m->a_arena.high / 1e6, m->s_arena.high / 1e6,
         (m->om_arena.high + m->ov_arena.high) / 1e6);

  free(idx); free(tgt);
  model_free(m);
  ref_free(r);
  return 0;
}
