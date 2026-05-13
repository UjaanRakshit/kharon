#include "flash.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <math.h>

template <class T> __device__ __forceinline__ float toF(T x);
template <> __device__ __forceinline__ float toF<float>(float x) { return x; }
template <> __device__ __forceinline__ float toF<__nv_bfloat16>(__nv_bfloat16 x) { return __bfloat162float(x); }
template <class T> __device__ __forceinline__ T fromF(float x);
template <> __device__ __forceinline__ float fromF<float>(float x) { return x; }
template <> __device__ __forceinline__ __nv_bfloat16 fromF<__nv_bfloat16>(float x) { return __float2bfloat16(x); }

// Warp-per-query-row forward. A warp owns one query row; its 32 lanes split the
// head dim (each lane holds hd/32 output elements in registers), dot-products
// reduce via shuffles. K/V tiles are staged in shared and reused by all warps.
#define F_WPB 8         // warps (query rows) per block
#define F_BC 32         // key tile
#define F_MAXHD 128
#define F_MAXELE (F_MAXHD / 32)

__device__ __forceinline__ float warp_sum(float v) {
  for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
  return __shfl_sync(0xffffffffu, v, 0);
}

template <class ST>
__global__ void flash_fwd_k(const ST *q, const ST *k, const ST *v,
                            ST *o, float *lse, int T, int hd, float scale) {
  __shared__ float Ks[F_BC][F_MAXHD], Vs[F_BC][F_MAXHD];
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int qr = blockIdx.x * F_WPB + warp, bh = blockIdx.y;
  long obase = (long)bh * T * hd;
  const ST *qb = q + obase, *kb = k + obase, *vb = v + obase;
  int ele = (hd + 31) / 32;

  float qreg[F_MAXELE], oreg[F_MAXELE];
  for (int c = 0; c < ele; c++) {
    int e = lane + c * 32;
    qreg[c] = (qr < T && e < hd) ? toF(qb[(long)qr * hd + e]) : 0.f;
    oreg[c] = 0.f;
  }
  float m = -INFINITY, l = 0.f;
  int maxqr = blockIdx.x * F_WPB + F_WPB - 1;
  for (int kt = 0; kt * F_BC <= maxqr && kt * F_BC < T; kt++) {
    for (int idx = threadIdx.x; idx < F_BC * hd; idx += blockDim.x) {
      int cc = idx / hd, e = idx % hd, kc = kt * F_BC + cc;
      Ks[cc][e] = kc < T ? toF(kb[(long)kc * hd + e]) : 0.f;
      Vs[cc][e] = kc < T ? toF(vb[(long)kc * hd + e]) : 0.f;
    }
    __syncthreads();
    if (qr < T) {
      for (int cc = 0; cc < F_BC; cc++) {
        int kc = kt * F_BC + cc;
        if (kc >= T || kc > qr) break;
        float part = 0.f;
        for (int c = 0; c < ele; c++) { int e = lane + c * 32; if (e < hd) part += qreg[c] * Ks[cc][e]; }
        float s = warp_sum(part) * scale;
        float m_new = fmaxf(m, s), corr = expf(m - m_new), p = expf(s - m_new);
        l = l * corr + p;
        for (int c = 0; c < ele; c++) { int e = lane + c * 32; if (e < hd) oreg[c] = oreg[c] * corr + p * Vs[cc][e]; }
        m = m_new;
      }
    }
    __syncthreads();
  }
  if (qr < T) {
    float inv = 1.f / l;
    for (int c = 0; c < ele; c++) { int e = lane + c * 32; if (e < hd) o[obase + (long)qr * hd + e] = fromF<ST>(oreg[c] * inv); }
    if (lane == 0) lse[(long)bh * T + qr] = m + logf(l);
  }
}

void flash_attn_fwd(const float *q, const float *k, const float *v,
                    float *o, float *lse, int B, int H, int T, int hd, float scale) {
  dim3 grid((T + F_WPB - 1) / F_WPB, B * H);
  flash_fwd_k<float><<<grid, F_WPB * 32>>>(q, k, v, o, lse, T, hd, scale);
}
void flash_attn_fwd_bf(const void *q, const void *k, const void *v,
                       void *o, float *lse, int B, int H, int T, int hd, float scale) {
  dim3 grid((T + F_WPB - 1) / F_WPB, B * H);
  flash_fwd_k<__nv_bfloat16><<<grid, F_WPB * 32>>>((const __nv_bfloat16 *)q,
      (const __nv_bfloat16 *)k, (const __nv_bfloat16 *)v, (__nv_bfloat16 *)o, lse, T, hd, scale);
}

// delta D[i] = sum_e dO[i,e] * O[i,e]  (per query row), needed by the backward.
__global__ void flash_delta_k(const float *dout, const float *o, float *D, int hd, long n) {
  long row = (long)blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= n) return;
  const float *d = dout + row * hd, *oo = o + row * hd;
  float s = 0.f;
  for (int e = 0; e < hd; e++) s += d[e] * oo[e];
  D[row] = s;
}

// dQ: warp per query row. Loops key tiles, recomputes P = exp(scale*q.k - lse),
// dS = P*(dO.v - D), accumulates dQ in registers (hd split across lanes). No race.
__global__ void flash_dq_k(const float *q, const float *k, const float *v,
                           const float *dout, const float *lse, const float *D,
                           float *dq, int T, int hd, float scale) {
  __shared__ float Ks[F_BC][F_MAXHD], Vs[F_BC][F_MAXHD];
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int qr = blockIdx.x * F_WPB + warp, bh = blockIdx.y;
  long base = (long)bh * T * hd;
  int ele = (hd + 31) / 32;
  float qreg[F_MAXELE], doreg[F_MAXELE], acc[F_MAXELE];
  for (int c = 0; c < ele; c++) {
    int e = lane + c * 32;
    qreg[c] = (qr < T && e < hd) ? q[base + (long)qr * hd + e] : 0.f;
    doreg[c] = (qr < T && e < hd) ? dout[base + (long)qr * hd + e] : 0.f;
    acc[c] = 0.f;
  }
  float Li = (qr < T) ? lse[(long)bh * T + qr] : 0.f, Di = (qr < T) ? D[(long)bh * T + qr] : 0.f;
  int maxqr = blockIdx.x * F_WPB + F_WPB - 1;
  for (int kt = 0; kt * F_BC <= maxqr && kt * F_BC < T; kt++) {
    for (int idx = threadIdx.x; idx < F_BC * hd; idx += blockDim.x) {
      int cc = idx / hd, e = idx % hd, kc = kt * F_BC + cc;
      Ks[cc][e] = kc < T ? k[base + (long)kc * hd + e] : 0.f;
      Vs[cc][e] = kc < T ? v[base + (long)kc * hd + e] : 0.f;
    }
    __syncthreads();
    if (qr < T) {
      for (int cc = 0; cc < F_BC; cc++) {
        int kc = kt * F_BC + cc;
        if (kc >= T || kc > qr) break;
        float sp = 0.f, dpp = 0.f;
        for (int c = 0; c < ele; c++) {
          int e = lane + c * 32;
          if (e < hd) { sp += qreg[c] * Ks[cc][e]; dpp += doreg[c] * Vs[cc][e]; }
        }
        float P = expf(warp_sum(sp) * scale - Li);
        float dS = P * (warp_sum(dpp) - Di);
        for (int c = 0; c < ele; c++) { int e = lane + c * 32; if (e < hd) acc[c] += scale * dS * Ks[cc][e]; }
      }
    }
    __syncthreads();
  }
  if (qr < T)
    for (int c = 0; c < ele; c++) { int e = lane + c * 32; if (e < hd) dq[base + (long)qr * hd + e] = acc[c]; }
}

// dK,dV: warp per key row. Loops query tiles (causal i>=j), accumulates dK,dV in
// registers (hd split across lanes). No race.
__global__ void flash_dkv_k(const float *q, const float *k, const float *v,
                            const float *dout, const float *lse, const float *D,
                            float *dk, float *dv, int T, int hd, float scale) {
  __shared__ float Qs[F_BC][F_MAXHD], dOs[F_BC][F_MAXHD], Ls[F_BC], Ds[F_BC];
  int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
  int kr = blockIdx.x * F_WPB + warp, bh = blockIdx.y;
  long base = (long)bh * T * hd;
  int ele = (hd + 31) / 32;
  float kreg[F_MAXELE], vreg[F_MAXELE], dka[F_MAXELE], dva[F_MAXELE];
  for (int c = 0; c < ele; c++) {
    int e = lane + c * 32;
    kreg[c] = (kr < T && e < hd) ? k[base + (long)kr * hd + e] : 0.f;
    vreg[c] = (kr < T && e < hd) ? v[base + (long)kr * hd + e] : 0.f;
    dka[c] = 0.f; dva[c] = 0.f;
  }
  int qt0 = (blockIdx.x * F_WPB) / F_BC, nqt = (T + F_BC - 1) / F_BC;
  for (int qt = qt0; qt < nqt; qt++) {
    for (int idx = threadIdx.x; idx < F_BC * hd; idx += blockDim.x) {
      int rr = idx / hd, e = idx % hd, i = qt * F_BC + rr;
      Qs[rr][e] = i < T ? q[base + (long)i * hd + e] : 0.f;
      dOs[rr][e] = i < T ? dout[base + (long)i * hd + e] : 0.f;
    }
    for (int rr = threadIdx.x; rr < F_BC; rr += blockDim.x) {
      int i = qt * F_BC + rr;
      Ls[rr] = i < T ? lse[(long)bh * T + i] : 0.f;
      Ds[rr] = i < T ? D[(long)bh * T + i] : 0.f;
    }
    __syncthreads();
    if (kr < T) {
      for (int rr = 0; rr < F_BC; rr++) {
        int i = qt * F_BC + rr;
        if (i >= T) break;
        if (i < kr) continue;
        float sp = 0.f, dpp = 0.f;
        for (int c = 0; c < ele; c++) {
          int e = lane + c * 32;
          if (e < hd) { sp += Qs[rr][e] * kreg[c]; dpp += dOs[rr][e] * vreg[c]; }
        }
        float P = expf(warp_sum(sp) * scale - Ls[rr]);
        float dS = P * (warp_sum(dpp) - Ds[rr]);
        for (int c = 0; c < ele; c++) {
          int e = lane + c * 32;
          if (e < hd) { dva[c] += P * dOs[rr][e]; dka[c] += scale * dS * Qs[rr][e]; }
        }
      }
    }
    __syncthreads();
  }
  if (kr < T)
    for (int c = 0; c < ele; c++) {
      int e = lane + c * 32;
      if (e < hd) { dk[base + (long)kr * hd + e] = dka[c]; dv[base + (long)kr * hd + e] = dva[c]; }
    }
}

void flash_attn_bwd(const float *q, const float *k, const float *v,
                    const float *o, const float *lse, const float *dout,
                    float *dq, float *dk, float *dv,
                    int B, int H, int T, int hd, float scale) {
  long n = (long)B * H * T;
  float *D;
  cudaMalloc((void **)&D, n * 4);
  flash_delta_k<<<(int)((n + 127) / 128), 128>>>(dout, o, D, hd, n);
  dim3 g((T + F_WPB - 1) / F_WPB, B * H);
  flash_dq_k<<<g, F_WPB * 32>>>(q, k, v, dout, lse, D, dq, T, hd, scale);
  flash_dkv_k<<<g, F_WPB * 32>>>(q, k, v, dout, lse, D, dk, dv, T, hd, scale);
  cudaFree(D);
}
