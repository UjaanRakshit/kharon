#include "flash.h"
#include "kernels.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>

// FlashAttention timing + the naive (materialized-scores) path it replaces.
// Reports TFLOP/s and FA-vs-naive speedup. FP32 this milestone; the strict
// %-of-official-FA (BF16) comparison is deferred to M3 when BF16 storage lands.
static float *q, *k, *v, *o, *lse, *att, *atto, *dq, *dk, *dv, *dout;
static int B, H, T, HD;
static float scale;

static float bench(const char *what, int iters) {
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  cudaDeviceSynchronize();
  cudaEventRecord(a, 0);
  for (int i = 0; i < iters; i++) {
    if (what[0] == 'F' && what[1] == 'F')      // FA fwd
      flash_attn_fwd(q, k, v, o, lse, B, H, T, HD, scale);
    else if (what[0] == 'F' && what[1] == 'B') {// FA fwd+bwd
      flash_attn_fwd(q, k, v, o, lse, B, H, T, HD, scale);
      flash_attn_bwd(q, k, v, o, lse, dout, dq, dk, dv, B, H, T, HD, scale);
    } else {                                    // naive fwd
      long sTT = (long)T * T, sThd = (long)T * HD;
      mm_nt_batched(q, k, att, T, T, HD, sThd, sThd, sTT, B * H);
      k_softmax_causal_fwd(att, B * H * T, T, scale);
      mm_nn_batched(att, v, atto, T, HD, T, sTT, sThd, sThd, B * H);
    }
  }
  cudaEventRecord(b, 0); cudaEventSynchronize(b);
  float ms = 0; cudaEventElapsedTime(&ms, a, b);
  cudaEventDestroy(a); cudaEventDestroy(b);
  return ms / iters;
}

int main(int argc, char **argv) {
  B = argc > 1 ? atoi(argv[1]) : 4;
  H = argc > 2 ? atoi(argv[2]) : 8;
  T = argc > 3 ? atoi(argv[3]) : 512;
  HD = argc > 4 ? atoi(argv[4]) : 64;
  scale = 1.f / sqrtf((float)HD);
  long N = (long)B * H * T * HD;
  gemm_init();
  CK(cudaMalloc((void **)&q, N * 4)); CK(cudaMalloc((void **)&k, N * 4)); CK(cudaMalloc((void **)&v, N * 4));
  CK(cudaMalloc((void **)&o, N * 4)); CK(cudaMalloc((void **)&atto, N * 4));
  CK(cudaMalloc((void **)&dq, N * 4)); CK(cudaMalloc((void **)&dk, N * 4)); CK(cudaMalloc((void **)&dv, N * 4));
  CK(cudaMalloc((void **)&dout, N * 4));
  CK(cudaMalloc((void **)&lse, (long)B * H * T * 4));
  CK(cudaMalloc((void **)&att, (long)B * H * T * T * 4));

  int it = 100;
  bench("FF", 10);                                  // warmup
  float ff = bench("FF", it), fb = bench("FB", it), nf = bench("NF", it);

  // causal attention FLOPs ~ 2 * (QK^T + AV) * 0.5 (triangular) = 2*B*H*T*T*hd
  double flop_fwd = 2.0 * B * H * (double)T * T * HD;
  double tf_ff = flop_fwd / (ff * 1e-3) / 1e12;
  printf("flash attention bench  B=%d H=%d T=%d hd=%d\n", B, H, T, HD);
  printf("  FA forward:      %.4f ms   %.1f TFLOP/s\n", ff, tf_ff);
  printf("  FA fwd+bwd:      %.4f ms\n", fb);
  printf("  naive forward:   %.4f ms\n", nf);
  printf("  FA vs naive fwd: %.2fx\n", nf / ff);
  gemm_destroy();
  return 0;
}
