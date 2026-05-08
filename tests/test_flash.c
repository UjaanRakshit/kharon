#include "refio.h"
#include "flash.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

#define RTOL 1e-4f
#define ATOL 1e-5f

static float *hbuf;
static long hcap;

static int cmp(const char *name, const float *dev, const float *ref, long n) {
  if (n > hcap) { hcap = n; hbuf = (float *)realloc(hbuf, n * 4); }
  CK(cudaMemcpy(hbuf, dev, n * 4, cudaMemcpyDeviceToHost));
  double maxabs = 0, maxrel = 0;
  long bad = 0;
  for (long i = 0; i < n; i++) {
    double diff = fabs((double)hbuf[i] - (double)ref[i]);
    if (diff > ATOL + RTOL * fabs((double)ref[i])) bad++;
    if (diff > maxabs) maxabs = diff;
    double rel = diff / (fabs((double)ref[i]) + 1e-12);
    if (rel > maxrel) maxrel = rel;
  }
  printf("  %-4s n=%-7ld maxabs=%.2e maxrel=%.2e bad=%ld -> %s\n",
         name, n, maxabs, maxrel, bad, bad ? "FAIL" : "ok");
  return bad == 0;
}

static float *dev_from(RefFile *r, const char *name, long n) {
  float *d;
  CK(cudaMalloc((void **)&d, n * 4));
  CK(cudaMemcpy(d, ref_f32(r, name), n * 4, cudaMemcpyHostToDevice));
  return d;
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "tests/attn_ref.bin";
  RefFile *r = ref_load(path);
  int B = r->batch, H = r->n_head, T = r->seq, hd = r->d_model / r->n_head;
  long N = (long)B * H * T * hd;
  float scale = 1.f / sqrtf((float)hd);
  printf("flash attention B=%d H=%d T=%d hd=%d\n", B, H, T, hd);

  float *q = dev_from(r, "q", N), *k = dev_from(r, "k", N), *v = dev_from(r, "v", N);
  float *o, *lse;
  CK(cudaMalloc((void **)&o, N * 4));
  CK(cudaMalloc((void **)&lse, (long)B * H * T * 4));

  flash_attn_fwd(q, k, v, o, lse, B, H, T, hd, scale);
  CK(cudaDeviceSynchronize());

  printf("forward:\n");
  int ok = cmp("o", o, ref_f32(r, "o"), N);

  free(hbuf);
  ref_free(r);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
