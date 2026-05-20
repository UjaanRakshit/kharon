#include "model.h"
#include "refio.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Single-GPU validation of the pipeline-stage path (P=1: one stage = full model,
// untied head). (1) grad accumulation: two backwards at inv_mb=0.5 == one at 1.0.
// (2) the stage trains (loss drops on a fixed batch). Multi-GPU 1F1B is on cluster.
int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "tests/m1_ref.bin";
  RefFile *r = ref_load(path);
  Config cfg = {r->n_layer, r->d_model, r->n_head, r->vocab, r->seq, r->batch};
  Model *m = model_create_pp(cfg, 1, 1);             // P=1: first and last
  model_init_weights_pp(m, 1337, 0, cfg.n_layer);
  model_set_input(m, ref_i32(r, "input_ids"), ref_i32(r, "targets"));
  long ng = m->g_arena.off / 4;
  float *gA = (float *)malloc(ng * 4), *gB = (float *)malloc(ng * 4);

  model_sync_bf16(m);
  model_zero_grads(m);
  model_forward_pp(m, NULL);
  model_backward_pp(m, NULL, 1.0f);
  CK(cudaMemcpy(gA, m->g_arena.base, ng * 4, cudaMemcpyDeviceToHost));

  model_zero_grads(m);
  model_forward_pp(m, NULL); model_backward_pp(m, NULL, 0.5f);
  model_forward_pp(m, NULL); model_backward_pp(m, NULL, 0.5f);
  CK(cudaMemcpy(gB, m->g_arena.base, ng * 4, cudaMemcpyDeviceToHost));

  double gmax = 0;
  for (long i = 0; i < ng; i++) { double dd = fabs((double)gA[i] - gB[i]); if (dd > gmax) gmax = dd; }
  int ok = gmax < 1e-3;
  printf("grad accumulation (2x0.5 vs 1x1.0): maxabs-diff %.3e -> %s\n", gmax, ok ? "ok" : "FAIL");

  float l0 = 0, lN = 0;
  for (int i = 0; i < 30; i++) {
    model_sync_bf16(m);
    model_zero_grads(m);
    float loss = model_forward_pp(m, NULL);
    model_backward_pp(m, NULL, 1.0f);
    model_adamw_step(m, 1e-2f, 0.9f, 0.95f, 1e-8f, 0.0f);
    if (i == 0) l0 = loss;
    if (i == 29) lN = loss;
  }
  printf("PP stage overfit: loss %.4f -> %.4f -> %s\n", l0, lN, lN < 0.5f * l0 ? "ok" : "FAIL");
  ok &= (lN < 0.5f * l0);

  free(gA); free(gB); model_free(m); ref_free(r);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
