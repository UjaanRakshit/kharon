#include "kernels.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// ZeRO-1 update-math oracle (single GPU, no NCCL): splitting the flat parameter buffer
// into dp contiguous shards and stepping AdamW per shard must equal one AdamW sweep over
// the whole buffer. This is the local stand-in for the sharded optimizer; the full-mesh
// reduce-scatter/all-gather is validated by the 8-GPU loss curve on the cluster.
// Also exercises padding (n not divisible by dp) so the tail never perturbs real params.
static void fill(float *h, long n, unsigned seed) {
  for (long i = 0; i < n; i++) { seed = seed * 1664525u + 1013904223u; h[i] = ((seed >> 8) / 8388608.0f - 1.0f) * 0.1f; }
}

static int run_case(long n, int dp, int steps) {
  long npad = ((n + dp - 1) / dp) * dp, shard = npad / dp;
  float b1 = 0.9f, b2 = 0.95f, eps = 1e-8f, wd = 0.1f, lr = 1e-3f;
  size_t pb = npad * 4, sb = shard * 4;
  float *pf, *gf, *mf, *vf;                          // full (reference) buffers
  float *ps, *gs, *ms, *vs;                          // sharded buffers
  CK(cudaMalloc(&pf, pb)); CK(cudaMalloc(&gf, pb)); CK(cudaMalloc(&mf, pb)); CK(cudaMalloc(&vf, pb));
  CK(cudaMalloc(&ps, pb)); CK(cudaMalloc(&gs, pb)); CK(cudaMalloc(&ms, pb)); CK(cudaMalloc(&vs, pb));
  float *hp = (float *)malloc(pb), *hg = (float *)malloc(pb);
  fill(hp, npad, 1); fill(hg, npad, 7);
  for (long i = n; i < npad; i++) { hp[i] = 0; hg[i] = 0; }   // zeroed padding tail
  CK(cudaMemcpy(pf, hp, pb, cudaMemcpyHostToDevice)); CK(cudaMemcpy(ps, hp, pb, cudaMemcpyHostToDevice));
  CK(cudaMemset(mf, 0, pb)); CK(cudaMemset(vf, 0, pb)); CK(cudaMemset(ms, 0, pb)); CK(cudaMemset(vs, 0, pb));

  for (int s = 1; s <= steps; s++) {
    float bc1 = 1.f - powf(b1, (float)s), bc2 = 1.f - powf(b2, (float)s);
    CK(cudaMemcpy(gf, hg, pb, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(gs, hg, pb, cudaMemcpyHostToDevice));
    k_adamw(pf, gf, mf, vf, npad, lr, b1, b2, eps, wd, bc1, bc2);            // unsharded
    for (int r = 0; r < dp; r++)                                            // sharded
      k_adamw(ps + (long)r * shard, gs + (long)r * shard, ms + (long)r * shard,
              vs + (long)r * shard, shard, lr, b1, b2, eps, wd, bc1, bc2);
  }
  CK(cudaDeviceSynchronize());
  float *a = (float *)malloc(pb), *b = (float *)malloc(pb);
  CK(cudaMemcpy(a, pf, pb, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(b, ps, pb, cudaMemcpyDeviceToHost));
  double mx = 0;
  for (long i = 0; i < n; i++) { double d = fabs((double)a[i] - b[i]); if (d > mx) mx = d; }
  int ok = mx == 0.0;       // elementwise + same bias-correction => bit-identical
  printf("  n=%ld dp=%d (npad=%ld shard=%ld) steps=%d: maxabs-diff %.3e -> %s\n",
         n, dp, npad, shard, steps, mx, ok ? "ok" : "FAIL");
  free(hp); free(hg); free(a); free(b);
  cudaFree(pf); cudaFree(gf); cudaFree(mf); cudaFree(vf);
  cudaFree(ps); cudaFree(gs); cudaFree(ms); cudaFree(vs);
  return ok;
}

int main(void) {
  int ok = 1;
  printf("ZeRO-1 sharded AdamW == unsharded AdamW:\n");
  ok &= run_case(1 << 16, 2, 20);     // even split, DP=2
  ok &= run_case(1 << 16, 4, 20);     // DP=4
  ok &= run_case(100003, 2, 20);      // odd count -> padding tail (DP=2)
  ok &= run_case(100003, 8, 15);      // odd count, DP=8 (the full mesh)
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
