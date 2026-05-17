#include "model.h"
#include "refio.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// (1) bf16 backward grads vs the verified fp32 backward grads (rel-Frobenius, bf16
// tol). (2) bf16 training overfits a fixed batch -> loss drops sharply.
int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "tests/m1_ref.bin";
  RefFile *r = ref_load(path);
  Config cfg = {r->n_layer, r->d_model, r->n_head, r->vocab, r->seq, r->batch};
  Model *m = model_create(cfg);
  model_load_ref(m, r);
  model_set_input(m, ref_i32(r, "input_ids"), ref_i32(r, "targets"));

  long ng = m->g_arena.off / 4;
  float *gf = (float *)malloc(ng * 4), *gb = (float *)malloc(ng * 4);

  // fp32 grads (reference)
  model_forward(m);
  model_backward(m);
  CK(cudaMemcpy(gf, m->g_arena.base, ng * 4, cudaMemcpyDeviceToHost));
  // bf16 grads
  model_sync_bf16(m);
  model_forward_bf16(m);
  model_backward_bf16(m);
  CK(cudaMemcpy(gb, m->g_arena.base, ng * 4, cudaMemcpyDeviceToHost));

  double num = 0, den = 0;
  for (long i = 0; i < ng; i++) { double dd = (double)gb[i] - gf[i]; num += dd * dd; den += (double)gf[i] * gf[i]; }
  double relfro = sqrt(num / den);
  printf("bf16 vs fp32 grads: rel-frobenius = %.3e (%ld params)\n", relfro, ng);
  int ok = relfro < 3e-2;

  // overfit a fixed batch with the bf16 training step
  float l0 = 0, lN = 0;
  int N = 30;
  for (int i = 0; i < N; i++) {
    model_sync_bf16(m);
    float loss = model_forward_bf16(m);
    model_backward_bf16(m);
    model_adamw_step(m, 1e-2f, 0.9f, 0.95f, 1e-8f, 0.0f);
    if (i == 0) l0 = loss;
    if (i == N - 1) lN = loss;
  }
  CK(cudaDeviceSynchronize());
  printf("bf16 overfit: loss %.4f -> %.4f over %d steps\n", l0, lN, N);
  ok &= (lN < 0.5f * l0);

  free(gf); free(gb); model_free(m); ref_free(r);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
