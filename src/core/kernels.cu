#include "kernels.h"
#include "kharon.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <math.h>

static cublasHandle_t g_h;

void gemm_init(void) {
  CUBLAS_CK(cublasCreate(&g_h));
  // deterministic, no tensor-core path that reorders accumulation
  CUBLAS_CK(cublasSetMathMode(g_h, CUBLAS_PEDANTIC_MATH));
}
void gemm_destroy(void) { cublasDestroy(g_h); }

// Row-major GEMM via column-major cuBLAS. A row-major matrix handed to cuBLAS is
// read as its transpose; the calls below are derived from that identity.
void mm_nt(const float *A, const float *B, float *C, int M, int N, int K) {
  const float a = 1.f, b = 0.f;
  CUBLAS_CK(cublasSgemm(g_h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
                        &a, B, K, A, K, &b, C, N));
}
void mm_nn(const float *A, const float *B, float *C, int M, int N, int K) {
  const float a = 1.f, b = 0.f;
  CUBLAS_CK(cublasSgemm(g_h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
                        &a, B, N, A, K, &b, C, N));
}
void mm_tn(const float *A, const float *B, float *C, int M, int N, int K) {
  const float a = 1.f, b = 0.f;
  CUBLAS_CK(cublasSgemm(g_h, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K,
                        &a, B, N, A, M, &b, C, N));
}
void mm_nt_batched(const float *A, const float *B, float *C, int M, int N, int K,
                   long sA, long sB, long sC, int nb) {
  const float a = 1.f, b = 0.f;
  CUBLAS_CK(cublasSgemmStridedBatched(g_h, CUBLAS_OP_T, CUBLAS_OP_N, N, M, K,
            &a, B, K, sB, A, K, sA, &b, C, N, sC, nb));
}
void mm_nn_batched(const float *A, const float *B, float *C, int M, int N, int K,
                   long sA, long sB, long sC, int nb) {
  const float a = 1.f, b = 0.f;
  CUBLAS_CK(cublasSgemmStridedBatched(g_h, CUBLAS_OP_N, CUBLAS_OP_N, N, M, K,
            &a, B, N, sB, A, K, sA, &b, C, N, sC, nb));
}

#define TPB 256
static inline int ndiv(long n, int b) { return (int)((n + b - 1) / b); }

__device__ float block_sum(float v, float *sh) {
  int t = threadIdx.x;
  sh[t] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s) sh[t] += sh[t + s];
    __syncthreads();
  }
  return sh[0];
}
__device__ float block_max(float v, float *sh) {
  int t = threadIdx.x;
  sh[t] = v;
  __syncthreads();
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (t < s) sh[t] = fmaxf(sh[t], sh[t + s]);
    __syncthreads();
  }
  return sh[0];
}

__global__ void embed_k(const float *wte, const float *wpe, const int *idx,
                        float *out, int T, int d, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int row = i / d, col = i % d, t = row % T, tok = idx[row];
  out[i] = wte[(long)tok * d + col] + wpe[(long)t * d + col];
}
void k_embed(const float *wte, const float *wpe, const int *idx, float *out,
             int B, int T, int d) {
  long n = (long)B * T * d;
  embed_k<<<ndiv(n, TPB), TPB>>>(wte, wpe, idx, out, T, d, n);
}

__global__ void layernorm_fwd_k(const float *x, const float *w, const float *b,
                                float *out, float *mean, float *rstd, int d) {
  int row = blockIdx.x;
  const float *xr = x + (long)row * d;
  float *outr = out + (long)row * d;
  extern __shared__ float sh[];
  float s = 0;
  for (int j = threadIdx.x; j < d; j += blockDim.x) s += xr[j];
  float mu = block_sum(s, sh) / d;
  float vs = 0;
  for (int j = threadIdx.x; j < d; j += blockDim.x) { float t = xr[j] - mu; vs += t * t; }
  float var = block_sum(vs, sh) / d;
  float rs = rsqrtf(var + 1e-5f);
  if (threadIdx.x == 0) { mean[row] = mu; rstd[row] = rs; }
  for (int j = threadIdx.x; j < d; j += blockDim.x)
    outr[j] = (xr[j] - mu) * rs * w[j] + b[j];
}
void k_layernorm_fwd(const float *x, const float *w, const float *b,
                     float *out, float *mean, float *rstd, int rows, int d) {
  layernorm_fwd_k<<<rows, TPB, TPB * sizeof(float)>>>(x, w, b, out, mean, rstd, d);
}

__global__ void add_bias_k(float *y, const float *b, int N, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  y[i] += b[i % N];
}
void k_add_bias(float *y, const float *b, int rows, int N) {
  long n = (long)rows * N;
  add_bias_k<<<ndiv(n, TPB), TPB>>>(y, b, N, n);
}

__global__ void split_heads_k(const float *qkv, float *q, float *k, float *v,
                              int T, int H, int hd, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int d = H * hd;
  int e = i % hd, t = (i / hd) % T, h = (i / ((long)hd * T)) % H, b = i / ((long)hd * T * H);
  long src = (long)(b * T + t) * 3 * d + h * hd + e;
  q[i] = qkv[src];
  k[i] = qkv[src + d];
  v[i] = qkv[src + 2 * d];
}
void k_split_heads(const float *qkv, float *q, float *k, float *v,
                   int B, int T, int H, int hd) {
  long n = (long)B * H * T * hd;
  split_heads_k<<<ndiv(n, TPB), TPB>>>(qkv, q, k, v, T, H, hd, n);
}

__global__ void merge_heads_k(const float *atto, float *out, int T, int H, int hd, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int d = H * hd;
  int e = i % hd, t = (i / hd) % T, h = (i / ((long)hd * T)) % H, b = i / ((long)hd * T * H);
  out[(long)(b * T + t) * d + h * hd + e] = atto[i];
}
void k_merge_heads(const float *atto, float *out, int B, int T, int H, int hd) {
  long n = (long)B * H * T * hd;
  merge_heads_k<<<ndiv(n, TPB), TPB>>>(atto, out, T, H, hd, n);
}

__global__ void softmax_causal_fwd_k(float *att, int T, float scale) {
  int row = blockIdx.x, t1 = row % T;
  float *a = att + (long)row * T;
  extern __shared__ float sh[];
  float m = -INFINITY;
  for (int j = threadIdx.x; j < T; j += blockDim.x)
    m = fmaxf(m, j <= t1 ? a[j] * scale : -INFINITY);
  m = block_max(m, sh);
  float s = 0;
  for (int j = threadIdx.x; j < T; j += blockDim.x) {
    float e = (j <= t1) ? expf(a[j] * scale - m) : 0.f;
    a[j] = e; s += e;
  }
  s = block_sum(s, sh);
  for (int j = threadIdx.x; j < T; j += blockDim.x) a[j] /= s;
}
void k_softmax_causal_fwd(float *att, int rows_bh, int T, float scale) {
  softmax_causal_fwd_k<<<rows_bh, TPB, TPB * sizeof(float)>>>(att, T, scale);
}

__device__ float gelu_f(float x) {
  const float c = 0.7978845608028654f;  // sqrt(2/pi)
  return 0.5f * x * (1.f + tanhf(c * (x + 0.044715f * x * x * x)));
}
__global__ void gelu_fwd_k(const float *x, float *y, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) y[i] = gelu_f(x[i]);
}
void k_gelu_fwd(const float *x, float *y, long n) {
  gelu_fwd_k<<<ndiv(n, TPB), TPB>>>(x, y, n);
}

__global__ void add_k(const float *a, const float *b, float *c, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) c[i] = a[i] + b[i];
}
void k_add(const float *a, const float *b, float *c, long n) {
  add_k<<<ndiv(n, TPB), TPB>>>(a, b, c, n);
}

__global__ void ce_fwd_k(const float *logits, const int *tgt, float *probs,
                         float *rowloss, int vocab) {
  int row = blockIdx.x;
  const float *lr = logits + (long)row * vocab;
  float *pr = probs + (long)row * vocab;
  extern __shared__ float sh[];
  float m = -INFINITY;
  for (int j = threadIdx.x; j < vocab; j += blockDim.x) m = fmaxf(m, lr[j]);
  m = block_max(m, sh);
  float s = 0;
  for (int j = threadIdx.x; j < vocab; j += blockDim.x) { float e = expf(lr[j] - m); pr[j] = e; s += e; }
  s = block_sum(s, sh);
  for (int j = threadIdx.x; j < vocab; j += blockDim.x) pr[j] /= s;
  if (threadIdx.x == 0) rowloss[row] = -logf(pr[tgt[row]]);
}
void k_cross_entropy_fwd(const float *logits, const int *tgt, float *probs,
                         float *rowloss, int rows, int vocab) {
  ce_fwd_k<<<rows, TPB, TPB * sizeof(float)>>>(logits, tgt, probs, rowloss, vocab);
}
