#include "kernels.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// Prove the M3 lever: BF16 tensor-core GEMM (cublasGemmEx) vs FP32 cuBLAS, on a
// transformer-shaped matmul. Reports speedup and bf16-vs-fp32 accuracy.
int main(int argc, char **argv) {
  int M = argc > 1 ? atoi(argv[1]) : 8192;   // rows (batch*seq)
  int N = argc > 2 ? atoi(argv[2]) : 8192;   // out features (e.g. 4*d)
  int K = argc > 3 ? atoi(argv[3]) : 2048;   // in features (d)
  long mn = (long)M * N, mk = (long)M * K, nk = (long)N * K;
  gemm_init();

  float *A, *B, *Cf, *Cb, *hb;
  void *Ab, *Bb;
  CK(cudaMalloc((void **)&A, mk * 4)); CK(cudaMalloc((void **)&B, nk * 4));
  CK(cudaMalloc((void **)&Cf, mn * 4)); CK(cudaMalloc((void **)&Cb, mn * 4));
  CK(cudaMalloc(&Ab, mk * 2)); CK(cudaMalloc(&Bb, nk * 2));
  hb = (float *)malloc(mn * 4);

  // fill A,B with small random values on host
  float *ha = (float *)malloc(mk * 4), *hbb = (float *)malloc(nk * 4);
  unsigned s = 12345;
  for (long i = 0; i < mk; i++) { s = s * 1664525u + 1013904223u; ha[i] = ((int)(s >> 9) / 4194304.f - 1.f) * 0.1f; }
  for (long i = 0; i < nk; i++) { s = s * 1664525u + 1013904223u; hbb[i] = ((int)(s >> 9) / 4194304.f - 1.f) * 0.1f; }
  CK(cudaMemcpy(A, ha, mk * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(B, hbb, nk * 4, cudaMemcpyHostToDevice));
  k_f2b(A, Ab, mk); k_f2b(B, Bb, nk);

  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  int it = 50;
  for (int i = 0; i < 10; i++) mm_nt(A, B, Cf, M, N, K);
  CK(cudaDeviceSynchronize());
  cudaEventRecord(a, 0); for (int i = 0; i < it; i++) mm_nt(A, B, Cf, M, N, K);
  cudaEventRecord(b, 0); cudaEventSynchronize(b);
  float msf = 0; cudaEventElapsedTime(&msf, a, b); msf /= it;

  for (int i = 0; i < 10; i++) mm_nt_bf16(Ab, Bb, Cb, M, N, K);
  CK(cudaDeviceSynchronize());
  cudaEventRecord(a, 0); for (int i = 0; i < it; i++) mm_nt_bf16(Ab, Bb, Cb, M, N, K);
  cudaEventRecord(b, 0); cudaEventSynchronize(b);
  float msb = 0; cudaEventElapsedTime(&msb, a, b); msb /= it;

  double flop = 2.0 * M * N * K;
  // accuracy: bf16 result vs fp32 result
  float *hf = (float *)malloc(mn * 4);
  CK(cudaMemcpy(hf, Cf, mn * 4, cudaMemcpyDeviceToHost));
  CK(cudaMemcpy(hb, Cb, mn * 4, cudaMemcpyDeviceToHost));
  double num = 0, den = 0;
  for (long i = 0; i < mn; i++) { double d = (double)hb[i] - hf[i]; num += d * d; den += (double)hf[i] * hf[i]; }
  double relfro = sqrt(num / den);
  printf("GEMM [%d x %d x %d]:\n", M, N, K);
  printf("  fp32 cuBLAS:  %.3f ms  %.1f TFLOP/s\n", msf, flop / (msf * 1e-3) / 1e12);
  printf("  bf16 TC:      %.3f ms  %.1f TFLOP/s\n", msb, flop / (msb * 1e-3) / 1e12);
  printf("  speedup:      %.2fx   bf16-vs-fp32 rel-frobenius=%.2e\n", msf / msb, relfro);
  gemm_destroy();
  return 0;
}
