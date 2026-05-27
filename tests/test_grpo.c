#include "model.h"
#include "grpo.h"
#include "refio.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// M9 oracles. (1) Group-relative advantages match the direct (r-mean)/std formula.
// (2) The GRPO policy-gradient backward is validated against the already-PyTorch-checked
// cross-entropy backward: with per-token coef = 1/R (mask=1, advantage=1, beta=0) the
// GRPO dlogits == the CE dlogits, so grads must be bit-identical; and coef scales the
// gradient linearly (coef=2/R -> 2x grads). This pins the whole policy-gradient path.

int main(int argc, char **argv) {
  int ok = 1;

  // (1) advantage normalization
  float rew[8] = {1, 0, 0, 1, 2, 2, 2, 2};   // group0 mixed, group1 all-equal (std 0)
  float adv[8];
  grpo_advantages(rew, 2, 4, 1e-4f, adv);
  // group0 mean=0.5 std=0.5 -> (0.5,-1,-1,0.5)/~0.5 ; group1 std=0 -> ~0
  double e0 = 0;
  float exp0[4] = {0.5f / 0.5001f, -0.5f / 0.5001f, -0.5f / 0.5001f, 0.5f / 0.5001f};
  for (int i = 0; i < 4; i++) e0 += fabs(adv[i] - exp0[i]);
  int a_ok = e0 < 1e-3 && fabs(adv[4]) < 1e-2 && fabs(adv[7]) < 1e-2;
  printf("advantages: g0=[%.3f %.3f %.3f %.3f] g1~0=[%.3f..] -> %s\n",
         adv[0], adv[1], adv[2], adv[3], adv[4], a_ok ? "ok" : "FAIL");
  ok &= a_ok;

  // (2) GRPO backward vs CE backward
  const char *path = argc > 1 ? argv[1] : "tests/m1_ref.bin";
  RefFile *r = ref_load(path);
  Config cfg = {r->n_layer, r->d_model, r->n_head, r->vocab, r->seq, r->batch};
  long R = (long)cfg.batch * cfg.seq;
  Model *m = model_create(cfg);
  model_load_ref(m, r);
  model_set_input(m, ref_i32(r, "input_ids"), ref_i32(r, "targets"));
  model_sync_bf16(m);
  long ng = m->g_arena.off / 4;
  float *gA = (float *)malloc(ng * 4), *gB = (float *)malloc(ng * 4), *gC = (float *)malloc(ng * 4);

  model_forward_bf16(m);
  model_backward_bf16(m);                                  // CE reference
  CK(cudaMemcpy(gA, m->g_arena.base, ng * 4, cudaMemcpyDeviceToHost));

  float *d_coef; CK(cudaMalloc(&d_coef, R * 4));
  float *h_coef = (float *)malloc(R * 4);
  for (long i = 0; i < R; i++) h_coef[i] = 1.f / R;        // adv=1, mask=1, beta=0
  CK(cudaMemcpy(d_coef, h_coef, R * 4, cudaMemcpyHostToDevice));
  model_forward_bf16(m);
  model_grpo_backward(m, d_coef);
  CK(cudaMemcpy(gB, m->g_arena.base, ng * 4, cudaMemcpyDeviceToHost));

  for (long i = 0; i < R; i++) h_coef[i] = 2.f / R;        // 2x -> grads should double
  CK(cudaMemcpy(d_coef, h_coef, R * 4, cudaMemcpyHostToDevice));
  model_forward_bf16(m);
  model_grpo_backward(m, d_coef);
  CK(cudaMemcpy(gC, m->g_arena.base, ng * 4, cudaMemcpyDeviceToHost));

  double dmax = 0, scale_err = 0, gmax = 0;
  for (long i = 0; i < ng; i++) {
    double d = fabs((double)gA[i] - gB[i]); if (d > dmax) dmax = d;
    double s = fabs((double)gC[i] - 2.0 * gA[i]); if (s > scale_err) scale_err = s;
    if (fabs((double)gA[i]) > gmax) gmax = fabs((double)gA[i]);
  }
  int id_ok = dmax == 0.0;             // bit-identical to CE backward
  int sc_ok = scale_err < 1e-3 * gmax + 1e-6;   // 2x coef -> 2x grad (bf16 rounding tol)
  printf("grpo-bwd vs CE-bwd: maxabs-diff %.3e -> %s\n", dmax, id_ok ? "ok" : "FAIL");
  printf("coef linearity (2x): maxabs(gC - 2*gA) %.3e -> %s\n", scale_err, sc_ok ? "ok" : "FAIL");
  ok &= id_ok && sc_ok;

  free(gA); free(gB); free(gC); free(h_coef); cudaFree(d_coef);
  model_free(m); ref_free(r);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
