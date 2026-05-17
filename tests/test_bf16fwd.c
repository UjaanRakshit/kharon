#include "model.h"
#include "refio.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// BF16 mixed-precision forward vs the PyTorch fp32 reference, at BF16 tolerance
// (CONVENTIONS: rtol=2e-2, atol=2e-2). Logits are bf16-stored, so compare loosely.
#define RTOL 2e-2f
#define ATOL 2e-2f

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "tests/m1_ref.bin";
  RefFile *r = ref_load(path);
  Config cfg = {r->n_layer, r->d_model, r->n_head, r->vocab, r->seq, r->batch};
  long R = (long)cfg.batch * cfg.seq, nlog = R * cfg.vocab;

  Model *m = model_create(cfg);
  model_load_ref(m, r);
  model_set_input(m, ref_i32(r, "input_ids"), ref_i32(r, "targets"));
  model_sync_bf16(m);
  float loss = model_forward_bf16(m);
  CK(cudaDeviceSynchronize());

  float ref_loss = *ref_f32(r, "loss");
  printf("bf16 forward: loss got=%.5f ref=%.5f (fp32) reldiff=%.3e\n",
         loss, ref_loss, fabs(loss - ref_loss) / ref_loss);

  // logits are bf16 in the arena; cast to fp32 for comparison
  void *dlog_bf = m->a_bf.logits;
  float *dlog_f;
  CK(cudaMalloc((void **)&dlog_f, nlog * 4));
  extern void k_b2f(const void *, float *, long);
  k_b2f(dlog_bf, dlog_f, nlog);
  float *got = (float *)malloc(nlog * 4);
  CK(cudaMemcpy(got, dlog_f, nlog * 4, cudaMemcpyDeviceToHost));
  const float *ref = ref_f32(r, "logits");
  double maxabs = 0;
  long bad = 0;
  for (long i = 0; i < nlog; i++) {
    double diff = fabs((double)got[i] - (double)ref[i]);
    if (diff > ATOL + RTOL * fabs((double)ref[i])) bad++;
    if (diff > maxabs) maxabs = diff;
  }
  int ok = (bad == 0) && (fabs(loss - ref_loss) <= ATOL + RTOL * fabs(ref_loss));
  printf("logits: n=%ld maxabs=%.3e bad=%ld -> %s\n", nlog, maxabs, bad, bad ? "FAIL" : "ok");

  free(got); model_free(m); ref_free(r);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
