#include "comms.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// 2-rank (or N-rank) NCCL all-reduce: correctness + PCIe bus bandwidth. This is
// the M4 interconnect measurement - L40S has no NVLink, so every byte is PCIe.
int main(int argc, char **argv) {
  Comms c;
  comms_init(&c);
  long maxn = 1L << 24;             // 64 MB of floats
  float *d;
  CK(cudaMalloc((void **)&d, maxn * 4));
  float *h = (float *)malloc(maxn * 4);

  // correctness at 1M floats: each rank contributes (rank+1); sum = nranks(nranks+1)/2
  long n = 1L << 20;
  for (long i = 0; i < n; i++) h[i] = (float)(c.rank + 1);
  CK(cudaMemcpy(d, h, n * 4, cudaMemcpyHostToDevice));
  comms_allreduce(&c, d, n);
  CK(cudaDeviceSynchronize());
  CK(cudaMemcpy(h, d, n * 4, cudaMemcpyDeviceToHost));
  float expect = c.nranks * (c.nranks + 1) / 2.0f;
  int ok = fabsf(h[0] - expect) < 1e-3f && fabsf(h[n - 1] - expect) < 1e-3f;
  if (c.rank == 0)
    printf("allreduce correctness (%d ranks): got %.1f expect %.1f -> %s\n",
           c.nranks, h[0], expect, ok ? "ok" : "FAIL");

  cudaEvent_t a, b;
  cudaEventCreate(&a); cudaEventCreate(&b);
  if (c.rank == 0) printf("%10s %10s %12s %12s\n", "size(MB)", "time(ms)", "algbw(GB/s)", "busbw(GB/s)");
  long sizes[] = {1L << 12, 1L << 16, 1L << 18, 1L << 20, 1L << 22, 1L << 24};
  for (int si = 0; si < (int)(sizeof(sizes) / sizeof(sizes[0])); si++) {
    long sz = sizes[si];
    for (int i = 0; i < 5; i++) comms_allreduce(&c, d, sz);   // warmup
    CK(cudaDeviceSynchronize());
    int iters = 30;
    CK(cudaEventRecord(a, 0));
    for (int i = 0; i < iters; i++) comms_allreduce(&c, d, sz);
    CK(cudaEventRecord(b, 0));
    CK(cudaEventSynchronize(b));
    float ms = 0; cudaEventElapsedTime(&ms, a, b); ms /= iters;
    double bytes = sz * 4.0;
    double algbw = bytes / 1e9 / (ms / 1e3);
    double busbw = algbw * 2.0 * (c.nranks - 1) / c.nranks;
    if (c.rank == 0)
      printf("%10.2f %10.4f %12.1f %12.1f\n", bytes / 1e6, ms, algbw, busbw);
  }
  free(h); cudaFree(d);
  comms_finalize(&c);
  return 0;
}
