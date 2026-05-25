#include "paged.h"
#include "kharon.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

// Inference kernels for the paged-KV decode path. Q/K/V are read straight from the
// fused qkv buffer [ntok,3d] (q=[..,0:d], k=[..,d:2d], v=[..,2d:3d]) viewed as
// [ntok,H,hd] — no head transpose, unlike the training FlashAttention layout.

template <class T> __device__ __forceinline__ float toF(T x);
template <> __device__ __forceinline__ float toF<float>(float x) { return x; }
template <> __device__ __forceinline__ float toF<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template <class T> __device__ __forceinline__ T fromF(float x);
template <> __device__ __forceinline__ float fromF<float>(float x) { return x; }
template <> __device__ __forceinline__ __nv_bfloat16 fromF<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

#define TPB 256
static inline int ndiv(long n, int b) { return (int)((n + b - 1) / b); }

// out[i,e] = wte[tok_i,e] + wpe[pos_i,e]   (per-token absolute position)
template <class T>
__global__ void embed_pos_k(const T *wte, const T *wpe, const int *tok, const int *pos,
                            T *out, int d, long n) {
  long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  long r = i / d, col = i % d;
  out[i] = fromF<T>(toF(wte[(long)tok[r] * d + col]) + toF(wpe[(long)pos[r] * d + col]));
}

// Scatter k,v for each token into its paged slot for layer `l`.
// cache layout per layer: [n_blocks, block_size, H, hd]; cache_l = base + l*blkstride.
template <class T>
__global__ void append_kv_k(const T *qkv, T *kc, T *vc, const int *seqslot, const int *pos,
                            const int *btab, int max_lb, int bs, int H, int hd, int d, long n) {
  long x = (long)blockIdx.x * blockDim.x + threadIdx.x;
  long tot = n * H * hd;
  if (x >= tot) return;
  int e = x % hd; int head = (x / hd) % H; long i = x / (hd * H);
  int p = pos[i], lb = p / bs, slot = p % bs;
  int pb = btab[(long)seqslot[i] * max_lb + lb];
  long dst = (((long)pb * bs + slot) * H + head) * hd + e;
  const T *qrow = qkv + i * (long)3 * d;
  kc[dst] = qrow[d + head * hd + e];
  vc[dst] = qrow[2 * d + head * hd + e];
}

// Causal paged attention. One block per (token, head); blockDim.x == hd. Online softmax
// over context positions 0..pos_i of the token's sequence. q read from qkv[i].
template <class T>
__global__ void paged_attn_k(const T *qkv, const T *kc, const T *vc, const int *seqslot,
                             const int *pos, const int *btab, int max_lb, int bs, int H,
                             int hd, int d, float scale, T *out) {
  int e = threadIdx.x;                 // dimension within head, e in [0,hd)
  int head = blockIdx.x % H;
  long i = blockIdx.x / H;
  extern __shared__ float sh[];        // [hd] reduction scratch
  const T *qrow = qkv + i * (long)3 * d;
  float q = toF(qrow[head * hd + e]);
  int P = pos[i];
  int slotrow = seqslot[i];
  float m = -1e30f, lsum = 0.f, acc = 0.f;
  for (int c = 0; c <= P; c++) {
    int lb = c / bs, slot = c % bs;
    int pb = btab[(long)slotrow * max_lb + lb];
    long off = (((long)pb * bs + slot) * H + head) * hd;
    float kv = q * toF(kc[off + e]);
    sh[e] = kv; __syncthreads();
    for (int s = hd >> 1; s > 0; s >>= 1) { if (e < s) sh[e] += sh[e + s]; __syncthreads(); }
    float score = sh[0] * scale; __syncthreads();
    float mnew = fmaxf(m, score), alpha = __expf(m - mnew), p = __expf(score - mnew);
    lsum = lsum * alpha + p;
    acc = acc * alpha + p * toF(vc[off + e]);
    m = mnew;
  }
  out[i * (long)d + head * hd + e] = fromF<T>(acc / lsum);
}

// Gather selected rows (last token of each sequence) from src[*,d] into dst[nrow,d].
template <class T>
__global__ void gather_rows_k(const T *src, const int *rows, T *dst, int d, long n) {
  long x = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (x >= n) return;
  int col = x % d; long r = x / d;
  dst[x] = src[(long)rows[r] * d + col];
}

// ---- launchers (fp32 + bf16) -------------------------------------------------
void k_embed_pos(const float *wte, const float *wpe, const int *tok, const int *pos,
                 float *out, int d, long ntok) {
  embed_pos_k<float><<<ndiv(ntok * d, TPB), TPB>>>(wte, wpe, tok, pos, out, d, ntok * d);
}
void k_embed_pos_bf(const void *wte, const void *wpe, const int *tok, const int *pos,
                    void *out, int d, long ntok) {
  embed_pos_k<__nv_bfloat16><<<ndiv(ntok * d, TPB), TPB>>>((const __nv_bfloat16 *)wte,
      (const __nv_bfloat16 *)wpe, tok, pos, (__nv_bfloat16 *)out, d, ntok * d);
}
void k_append_kv(const float *qkv, float *kc, float *vc, const int *seqslot, const int *pos,
                 const int *btab, int max_lb, int bs, int H, int hd, int d, long ntok) {
  long tot = ntok * H * hd;
  append_kv_k<float><<<ndiv(tot, TPB), TPB>>>(qkv, kc, vc, seqslot, pos, btab, max_lb, bs, H, hd, d, ntok);
}
void k_append_kv_bf(const void *qkv, void *kc, void *vc, const int *seqslot, const int *pos,
                    const int *btab, int max_lb, int bs, int H, int hd, int d, long ntok) {
  long tot = ntok * H * hd;
  append_kv_k<__nv_bfloat16><<<ndiv(tot, TPB), TPB>>>((const __nv_bfloat16 *)qkv,
      (__nv_bfloat16 *)kc, (__nv_bfloat16 *)vc, seqslot, pos, btab, max_lb, bs, H, hd, d, ntok);
}
void k_paged_attn(const float *qkv, const float *kc, const float *vc, const int *seqslot,
                  const int *pos, const int *btab, int max_lb, int bs, int H, int hd, int d,
                  float scale, float *out, long ntok) {
  paged_attn_k<float><<<ntok * H, hd, hd * sizeof(float)>>>(qkv, kc, vc, seqslot, pos, btab,
      max_lb, bs, H, hd, d, scale, out);
}
void k_paged_attn_bf(const void *qkv, const void *kc, const void *vc, const int *seqslot,
                     const int *pos, const int *btab, int max_lb, int bs, int H, int hd, int d,
                     float scale, void *out, long ntok) {
  paged_attn_k<__nv_bfloat16><<<ntok * H, hd, hd * sizeof(float)>>>((const __nv_bfloat16 *)qkv,
      (const __nv_bfloat16 *)kc, (const __nv_bfloat16 *)vc, seqslot, pos, btab, max_lb, bs, H,
      hd, d, scale, (__nv_bfloat16 *)out);
}
void k_gather_rows(const float *src, const int *rows, float *dst, int d, long nrow) {
  gather_rows_k<float><<<ndiv(nrow * d, TPB), TPB>>>(src, rows, dst, d, nrow * d);
}
void k_gather_rows_bf(const void *src, const int *rows, void *dst, int d, long nrow) {
  gather_rows_k<__nv_bfloat16><<<ndiv(nrow * d, TPB), TPB>>>((const __nv_bfloat16 *)src, rows,
      (__nv_bfloat16 *)dst, d, nrow * d);
}
