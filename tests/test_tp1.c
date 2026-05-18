#include "model.h"
#include "refio.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// At tp=1 (all-reduce is identity), the tensor-parallel path must equal the
// single-GPU bf16 path bit-for-bit. Validates the TP fwd/bwd wiring without GPUs.
int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "tests/m1_ref.bin";
  RefFile *r = ref_load(path);
  Config cfg = {r->n_layer, r->d_model, r->n_head, r->vocab, r->seq, r->batch};
  Model *m = model_create_tp(cfg, 1, 0);      // tp=1, allreduce stays NULL
  model_load_ref(m, r);
  model_set_input(m, ref_i32(r, "input_ids"), ref_i32(r, "targets"));
  long ng = m->g_arena.off / 4;
  float *gA = (float *)malloc(ng * 4), *gB = (float *)malloc(ng * 4);

  model_sync_bf16(m);
  float la = model_forward_bf16(m);
  model_backward_bf16(m);
  CK(cudaMemcpy(gA, m->g_arena.base, ng * 4, cudaMemcpyDeviceToHost));

  float lb = model_forward_tp(m);
  model_backward_tp(m);
  CK(cudaMemcpy(gB, m->g_arena.base, ng * 4, cudaMemcpyDeviceToHost));

  double gmax = 0;
  for (long i = 0; i < ng; i++) { double dd = fabs((double)gA[i] - gB[i]); if (dd > gmax) gmax = dd; }
  int ok = (la == lb) && (gmax == 0.0);
  printf("tp=1 vs bf16: loss %.6f vs %.6f; grad maxabs-diff %.3e -> %s\n", la, lb, gmax, ok ? "PASS" : "FAIL");

  free(gA); free(gB); model_free(m); ref_free(r);
  return ok ? 0 : 1;
}
