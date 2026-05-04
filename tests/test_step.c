#include "model.h"
#include "refio.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define RTOL 1e-4f
#define ATOL 1e-5f

static float *hbuf;
static long hcap;

static int cmp_dev(RefFile *r, const char *name, const float *dptr, long n) {
  if (n > hcap) { hcap = n; hbuf = (float *)realloc(hbuf, n * 4); }
  CK(cudaMemcpy(hbuf, dptr, n * 4, cudaMemcpyDeviceToHost));
  const float *ref = ref_f32(r, name);
  double maxabs = 0;
  long bad = 0;
  for (long i = 0; i < n; i++) {
    double diff = fabs((double)hbuf[i] - (double)ref[i]);
    if (diff > ATOL + RTOL * fabs((double)ref[i])) bad++;
    if (diff > maxabs) maxabs = diff;
  }
  printf("  %-18s n=%-7ld maxabs=%.2e bad=%ld -> %s\n", name, n, maxabs, bad, bad ? "FAIL" : "ok");
  return bad == 0;
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "tests/m1_ref.bin";
  RefFile *r = ref_load(path);
  Config cfg = {r->n_layer, r->d_model, r->n_head, r->vocab, r->seq, r->batch};
  int d = cfg.d_model, ff = 4 * d;
  long V = cfg.vocab, S = cfg.seq;

  Model *m = model_create(cfg);
  model_load_ref(m, r);
  model_set_input(m, ref_i32(r, "input_ids"), ref_i32(r, "targets"));
  model_forward(m);
  model_backward(m);
  model_adamw_step(m, r->lr, r->beta1, r->beta2, r->eps, r->wd);
  CK(cudaDeviceSynchronize());

  printf("adamw step:\n");
  int ok = 1;
  ok &= cmp_dev(r, "wte.step", m->w.wte, V * d);
  ok &= cmp_dev(r, "wpe.step", m->w.wpe, S * d);
  char nm[64];
  for (int l = 0; l < cfg.n_layer; l++) {
    LayerW *w = &m->w.layer[l];
#define C(field, suffix, cnt) do { snprintf(nm, sizeof(nm), "blk%d." suffix ".step", l); ok &= cmp_dev(r, nm, w->field, cnt); } while (0)
    C(ln1_w, "ln1.w", d); C(ln1_b, "ln1.b", d);
    C(qkv_w, "qkv.w", (long)3 * d * d); C(qkv_b, "qkv.b", 3 * d);
    C(proj_w, "proj.w", (long)d * d); C(proj_b, "proj.b", d);
    C(ln2_w, "ln2.w", d); C(ln2_b, "ln2.b", d);
    C(fc_w, "fc.w", (long)ff * d); C(fc_b, "fc.b", ff);
    C(fcproj_w, "fcproj.w", (long)d * ff); C(fcproj_b, "fcproj.b", d);
#undef C
  }
  ok &= cmp_dev(r, "lnf.w.step", m->w.lnf_w, d);
  ok &= cmp_dev(r, "lnf.b.step", m->w.lnf_b, d);

  free(hbuf);
  model_free(m);
  ref_free(r);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
