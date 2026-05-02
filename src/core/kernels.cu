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

void mm_tn_batched(const float *A, const float *B, float *C, int M, int N, int K,
                   long sA, long sB, long sC, int nb) {
  const float a = 1.f, b = 0.f;
  CUBLAS_CK(cublasSgemmStridedBatched(g_h, CUBLAS_OP_N, CUBLAS_OP_T, N, M, K,
            &a, B, N, sB, A, M, sA, &b, C, N, sC, nb));
}

#define TPB 256
static inline int ndiv(long n, int b) { return (int)((n + b - 1) / b); }

__device__ float block_sum(float v, float *sh) {
  int t = threadIdx.x;
  __syncthreads();          // ensure prior readers of sh are done before reuse
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
  __syncthreads();          // ensure prior readers of sh are done before reuse
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

// ---- backward ----------------------------------------------------------------
__global__ void ce_bwd_k(const float *probs, const int *tgt, float *dlogits,
                         int vocab, float invN) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long n = (long)gridDim.x * blockDim.x;
  (void)n;
  int row = i / vocab, col = i % vocab;
  dlogits[i] = (probs[i] - (col == tgt[row] ? 1.f : 0.f)) * invN;
}
void k_cross_entropy_bwd(const float *probs, const int *tgt, float *dlogits,
                         int rows, int vocab, float invN) {
  long n = (long)rows * vocab;
  ce_bwd_k<<<ndiv(n, TPB), TPB>>>(probs, tgt, dlogits, vocab, invN);
}

__global__ void layernorm_bwd_dx_k(const float *dy, const float *x, const float *w,
                                   const float *mean, const float *rstd, float *dx, int d) {
  int row = blockIdx.x;
  const float *dyr = dy + (long)row * d, *xr = x + (long)row * d;
  float *dxr = dx + (long)row * d;
  float mu = mean[row], rs = rstd[row];
  extern __shared__ float sh[];
  float s1 = 0, s2 = 0;
  for (int j = threadIdx.x; j < d; j += blockDim.x) {
    float xh = (xr[j] - mu) * rs, dxh = dyr[j] * w[j];
    s1 += dxh; s2 += dxh * xh;
  }
  float m1 = block_sum(s1, sh) / d;
  float m2 = block_sum(s2, sh) / d;
  for (int j = threadIdx.x; j < d; j += blockDim.x) {
    float xh = (xr[j] - mu) * rs, dxh = dyr[j] * w[j];
    dxr[j] = rs * (dxh - m1 - xh * m2);
  }
}
// one thread per column j: deterministic reduction over rows for dw, db
__global__ void layernorm_bwd_dwdb_k(const float *dy, const float *x,
                                     const float *mean, const float *rstd,
                                     float *dw, float *db, int rows, int d) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= d) return;
  float sw = 0, sb = 0;
  for (int r = 0; r < rows; r++) {
    float dyv = dy[(long)r * d + j];
    float xh = (x[(long)r * d + j] - mean[r]) * rstd[r];
    sw += dyv * xh; sb += dyv;
  }
  dw[j] += sw; db[j] += sb;
}
void k_layernorm_bwd(const float *dy, const float *x, const float *w,
                     const float *mean, const float *rstd,
                     float *dx, float *dw, float *db, int rows, int d) {
  layernorm_bwd_dx_k<<<rows, TPB, TPB * sizeof(float)>>>(dy, x, w, mean, rstd, dx, d);
  layernorm_bwd_dwdb_k<<<ndiv(d, TPB), TPB>>>(dy, x, mean, rstd, dw, db, rows, d);
}

__global__ void gelu_bwd_k(const float *x, const float *dy, float *dx, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const float c = 0.7978845608028654f, a = 0.044715f;
  float xv = x[i];
  float u = c * (xv + a * xv * xv * xv);
  float t = tanhf(u);
  float dudx = c * (1.f + 3.f * a * xv * xv);
  float dg = 0.5f * (1.f + t) + 0.5f * xv * (1.f - t * t) * dudx;
  dx[i] = dy[i] * dg;
}
void k_gelu_bwd(const float *x, const float *dy, float *dx, long n) {
  gelu_bwd_k<<<ndiv(n, TPB), TPB>>>(x, dy, dx, n);
}

__global__ void softmax_causal_bwd_k(const float *att, const float *datt,
                                     float *dscores, int T, float scale) {
  int row = blockIdx.x, t1 = row % T;
  const float *a = att + (long)row * T, *da = datt + (long)row * T;
  float *ds = dscores + (long)row * T;
  extern __shared__ float sh[];
  float s = 0;
  for (int j = threadIdx.x; j < T; j += blockDim.x) s += (j <= t1) ? a[j] * da[j] : 0.f;
  s = block_sum(s, sh);
  for (int j = threadIdx.x; j < T; j += blockDim.x)
    ds[j] = (j <= t1) ? a[j] * (da[j] - s) * scale : 0.f;
}
void k_softmax_causal_bwd(const float *att, const float *datt, float *dscores,
                          int rows_bh, int T, float scale) {
  softmax_causal_bwd_k<<<rows_bh, TPB, TPB * sizeof(float)>>>(att, datt, dscores, T, scale);
}

__global__ void unmerge_heads_k(const float *dout, float *datto, int T, int H, int hd, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int d = H * hd;
  int e = i % hd, t = (i / hd) % T, h = (i / ((long)hd * T)) % H, b = i / ((long)hd * T * H);
  datto[i] = dout[(long)(b * T + t) * d + h * hd + e];
}
void k_unmerge_heads(const float *dout, float *datto, int B, int T, int H, int hd) {
  long n = (long)B * H * T * hd;
  unmerge_heads_k<<<ndiv(n, TPB), TPB>>>(dout, datto, T, H, hd, n);
}

__global__ void combine_qkv_k(const float *dq, const float *dk, const float *dv,
                              float *dqkv, int T, int H, int hd, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int d = H * hd;
  int e = i % hd, t = (i / hd) % T, h = (i / ((long)hd * T)) % H, b = i / ((long)hd * T * H);
  long dst = (long)(b * T + t) * 3 * d + h * hd + e;
  dqkv[dst] = dq[i];
  dqkv[dst + d] = dk[i];
  dqkv[dst + 2 * d] = dv[i];
}
void k_combine_qkv(const float *dq, const float *dk, const float *dv, float *dqkv,
                   int B, int T, int H, int hd) {
  long n = (long)B * H * T * hd;
  combine_qkv_k<<<ndiv(n, TPB), TPB>>>(dq, dk, dv, dqkv, T, H, hd, n);
}

__global__ void colsum_k(const float *in, float *out, int rows, int N) {
  int j = blockIdx.x * blockDim.x + threadIdx.x;
  if (j >= N) return;
  float s = 0;
  for (int r = 0; r < rows; r++) s += in[(long)r * N + j];
  out[j] += s;
}
void k_colsum(const float *in, float *out, int rows, int N) {
  colsum_k<<<ndiv(N, TPB), TPB>>>(in, out, rows, N);
}

__global__ void embed_bwd_wte_k(const float *demb, const int *idx, float *dwte,
                                int rows, int d, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int v = i / d, e = i % d;
  float s = 0;
  for (int r = 0; r < rows; r++) if (idx[r] == v) s += demb[(long)r * d + e];
  dwte[i] += s;
}
void k_embed_bwd_wte(const float *demb, const int *idx, float *dwte,
                     int rows, int vocab, int d) {
  long n = (long)vocab * d;
  embed_bwd_wte_k<<<ndiv(n, TPB), TPB>>>(demb, idx, dwte, rows, d, n);
}

__global__ void embed_bwd_wpe_k(const float *demb, float *dwpe, int B, int T, int d, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  int t = i / d, e = i % d;
  float s = 0;
  for (int b = 0; b < B; b++) s += demb[(long)(b * T + t) * d + e];
  dwpe[i] += s;
}
void k_embed_bwd_wpe(const float *demb, float *dwpe, int B, int T, int d) {
  long n = (long)T * d;
  embed_bwd_wpe_k<<<ndiv(n, TPB), TPB>>>(demb, dwpe, B, T, d, n);
}
