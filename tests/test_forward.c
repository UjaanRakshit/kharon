#include "model.h"
#include "refio.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define RTOL 1e-4f
#define ATOL 1e-5f

static int cmp(const char *name, const float *got, const float *ref, long n) {
  double maxabs = 0, maxrel = 0;
  long bad = 0;
  for (long i = 0; i < n; i++) {
    double diff = fabs((double)got[i] - (double)ref[i]);
    double tol = ATOL + RTOL * fabs((double)ref[i]);
    if (diff > tol) bad++;
    if (diff > maxabs) maxabs = diff;
    double rel = diff / (fabs((double)ref[i]) + 1e-12);
    if (rel > maxrel) maxrel = rel;
  }
  printf("  %-10s n=%-8ld maxabs=%.3e maxrel=%.3e bad=%ld -> %s\n",
         name, n, maxabs, maxrel, bad, bad ? "FAIL" : "ok");
  return bad == 0;
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "tests/m1_ref.bin";
  RefFile *r = ref_load(path);
  Config cfg = {r->n_layer, r->d_model, r->n_head, r->vocab, r->seq, r->batch};
  long R = (long)cfg.batch * cfg.seq;

  Model *m = model_create(cfg);
  model_load_ref(m, r);
  model_set_input(m, ref_i32(r, "input_ids"), ref_i32(r, "targets"));
  float loss = model_forward(m);
  CK(cudaDeviceSynchronize());

  printf("forward:\n");
  int ok = 1;
  float ref_loss = *ref_f32(r, "loss");
  printf("  loss got=%.6f ref=%.6f diff=%.3e\n", loss, ref_loss, fabs(loss - ref_loss));
  ok &= fabs(loss - ref_loss) <= ATOL + RTOL * fabs(ref_loss);

  long nlog = R * cfg.vocab;
  float *logits = (float *)malloc(nlog * 4);
  CK(cudaMemcpy(logits, m->a.logits, nlog * 4, cudaMemcpyDeviceToHost));
  ok &= cmp("logits", logits, ref_f32(r, "logits"), nlog);

  free(logits);
  model_free(m);
  ref_free(r);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
