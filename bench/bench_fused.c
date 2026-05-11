#include "kernels.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// Fused vs unfused elementwise block ops. These are memory-bound, so fusing two
// passes into one should roughly halve traffic. Reports ms/iter and speedup.
static float time_ms(void (*fn)(void *), void *ctx, int iters) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int i = 0; i < 10; i++) fn(ctx);          // warmup
  cudaDeviceSynchronize();
  cudaEventRecord(a, 0);
  for (int i = 0; i < iters; i++) fn(ctx);
  cudaEventRecord(b, 0); cudaEventSynchronize(b);
  float ms = 0; cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a); cudaEventDestroy(b);
  return ms / iters;
}

static float *Y, *BIAS, *RES, *OUT, *PRE, *ACT;
static int ROWS, N;

static void unfused_bias_res(void *_) { (void)_; k_add_bias(OUT, BIAS, ROWS, N); k_add(RES, OUT, OUT, (long)ROWS * N); }
static void fused_bias_res(void *_)   { (void)_; k_bias_residual(Y, BIAS, RES, OUT, ROWS, N); }
static void unfused_bias_gelu(void *_){ (void)_; k_add_bias(OUT, BIAS, ROWS, N); k_gelu_fwd(OUT, ACT, (long)ROWS * N); }
static void fused_bias_gelu(void *_)  { (void)_; k_bias_gelu(Y, BIAS, PRE, ACT, ROWS, N); }

int main(int argc, char **argv) {
  ROWS = argc > 1 ? atoi(argv[1]) : 8192;
  N = argc > 2 ? atoi(argv[2]) : 2048;
  long n = (long)ROWS * N;
  CK(cudaMalloc((void **)&Y, n * 4)); CK(cudaMalloc((void **)&BIAS, N * 4));
  CK(cudaMalloc((void **)&RES, n * 4)); CK(cudaMalloc((void **)&OUT, n * 4));
  CK(cudaMalloc((void **)&PRE, n * 4)); CK(cudaMalloc((void **)&ACT, n * 4));

  int it = 500;
  float u1 = time_ms(unfused_bias_res, 0, it), f1 = time_ms(fused_bias_res, 0, it);
  float u2 = time_ms(unfused_bias_gelu, 0, it), f2 = time_ms(fused_bias_gelu, 0, it);
  printf("elementwise fusion [%d x %d]:\n", ROWS, N);
  printf("  bias+residual: unfused %.4f ms  fused %.4f ms  speedup %.2fx\n", u1, f1, u1 / f1);
  printf("  bias+gelu:     unfused %.4f ms  fused %.4f ms  speedup %.2fx\n", u2, f2, u2 / f2);
  return 0;
}
