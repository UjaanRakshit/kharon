#include "flash.h"
#include <cuda_runtime.h>
#include <math.h>

// Tile sizes tuned for sm_89. Static shared = (2*BR + 2*BC)*MAXHD*4 bytes; with
// these values that is 32 KB, under the 48 KB default. hd<=MAXHD (asserted by the
// host launcher). Larger head dims (1B model, hd=128) will move to dynamic shared.
#define BR 32
#define BC 32
#define MAXHD 64

// One block per (query tile, b*h). One thread per query row in the tile.
__global__ void flash_fwd_k(const float *q, const float *k, const float *v,
                            float *o, float *lse, int T, int hd, float scale) {
  __shared__ float Qs[BR][MAXHD], Os[BR][MAXHD], Ks[BC][MAXHD], Vs[BC][MAXHD];
  int t = threadIdx.x;
  int qtile = blockIdx.x, bh = blockIdx.y;
  int qr = qtile * BR + t;                 // global query row this thread owns
  const float *base = q + (long)bh * T * hd;
  const float *kbase = k + (long)bh * T * hd, *vbase = v + (long)bh * T * hd;

  if (qr < T)
    for (int e = 0; e < hd; e++) { Qs[t][e] = base[(long)qr * hd + e]; Os[t][e] = 0.f; }
  float m = -INFINITY, l = 0.f;
  __syncthreads();

  int maxqr = qtile * BR + BR - 1;         // causal: skip key tiles fully in the future
  for (int kt = 0; kt * BC <= maxqr && kt * BC < T; kt++) {
    for (int idx = t; idx < BC * hd; idx += BR) {
      int c = idx / hd, e = idx % hd, kc = kt * BC + c;
      Ks[c][e] = kc < T ? kbase[(long)kc * hd + e] : 0.f;
      Vs[c][e] = kc < T ? vbase[(long)kc * hd + e] : 0.f;
    }
    __syncthreads();
    if (qr < T) {
      for (int c = 0; c < BC; c++) {
        int kc = kt * BC + c;
        if (kc >= T || kc > qr) break;     // causal + bounds (cols are in order)
        float s = 0.f;
        for (int e = 0; e < hd; e++) s += Qs[t][e] * Ks[c][e];
        s *= scale;
        float m_new = fmaxf(m, s);
        float corr = expf(m - m_new), p = expf(s - m_new);
        l = l * corr + p;
        for (int e = 0; e < hd; e++) Os[t][e] = Os[t][e] * corr + p * Vs[c][e];
        m = m_new;
      }
    }
    __syncthreads();
  }
  if (qr < T) {
    float inv = 1.f / l;
    for (int e = 0; e < hd; e++) o[(long)bh * T * hd + (long)qr * hd + e] = Os[t][e] * inv;
    lse[(long)bh * T + qr] = m + logf(l);
  }
}

void flash_attn_fwd(const float *q, const float *k, const float *v,
                    float *o, float *lse, int B, int H, int T, int hd, float scale) {
  dim3 grid((T + BR - 1) / BR, B * H);
  flash_fwd_k<<<grid, BR>>>(q, k, v, o, lse, T, hd, scale);
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

// dQ: one block per (query tile, bh), one thread per query row. Loops key tiles,
// recomputes P = exp(scale*q.k - lse), dS = P*(dO.v - D), accumulates dQ. No race.
__global__ void flash_dq_k(const float *q, const float *k, const float *v,
                           const float *dout, const float *lse, const float *D,
                           float *dq, int T, int hd, float scale) {
  __shared__ float Qs[BR][MAXHD], dOs[BR][MAXHD], Ks[BC][MAXHD], Vs[BC][MAXHD];
  int t = threadIdx.x, qtile = blockIdx.x, bh = blockIdx.y, qr = qtile * BR + t;
  long base = (long)bh * T * hd;
  float acc[MAXHD];
  for (int e = 0; e < hd; e++) acc[e] = 0.f;
  float Li = 0, Di = 0;
  if (qr < T) {
    for (int e = 0; e < hd; e++) { Qs[t][e] = q[base + (long)qr * hd + e]; dOs[t][e] = dout[base + (long)qr * hd + e]; }
    Li = lse[(long)bh * T + qr]; Di = D[(long)bh * T + qr];
  }
  __syncthreads();
  int maxqr = qtile * BR + BR - 1;
  for (int kt = 0; kt * BC <= maxqr && kt * BC < T; kt++) {
    for (int idx = t; idx < BC * hd; idx += BR) {
      int c = idx / hd, e = idx % hd, kc = kt * BC + c;
      Ks[c][e] = kc < T ? k[base + (long)kc * hd + e] : 0.f;
      Vs[c][e] = kc < T ? v[base + (long)kc * hd + e] : 0.f;
    }
    __syncthreads();
    if (qr < T) {
      for (int c = 0; c < BC; c++) {
        int kc = kt * BC + c;
        if (kc >= T || kc > qr) break;
        float s = 0.f;
        for (int e = 0; e < hd; e++) s += Qs[t][e] * Ks[c][e];
        float P = expf(s * scale - Li);
        float dP = 0.f;
        for (int e = 0; e < hd; e++) dP += dOs[t][e] * Vs[c][e];
        float dS = P * (dP - Di);
        for (int e = 0; e < hd; e++) acc[e] += scale * dS * Ks[c][e];
      }
    }
    __syncthreads();
  }
  if (qr < T)
    for (int e = 0; e < hd; e++) dq[base + (long)qr * hd + e] = acc[e];
}

// dK,dV: one block per (key tile, bh), one thread per key row. Loops query tiles
// (causal i>=j), accumulates dK,dV in registers. No race.
__global__ void flash_dkv_k(const float *q, const float *k, const float *v,
                            const float *dout, const float *lse, const float *D,
                            float *dk, float *dv, int T, int hd, float scale) {
  __shared__ float Ks[BC][MAXHD], Vs[BC][MAXHD], Qs[BR][MAXHD], dOs[BR][MAXHD];
  __shared__ float Ls[BR], Ds[BR];
  int t = threadIdx.x, ktile = blockIdx.x, bh = blockIdx.y, kr = ktile * BC + t;
  long base = (long)bh * T * hd;
  float dKa[MAXHD], dVa[MAXHD];
  for (int e = 0; e < hd; e++) { dKa[e] = 0.f; dVa[e] = 0.f; }
  if (kr < T)
    for (int e = 0; e < hd; e++) { Ks[t][e] = k[base + (long)kr * hd + e]; Vs[t][e] = v[base + (long)kr * hd + e]; }
  __syncthreads();
  int qt0 = (ktile * BC) / BR;             // first query tile that can be causal
  int nqt = (T + BR - 1) / BR;
  for (int qt = qt0; qt < nqt; qt++) {
    for (int idx = t; idx < BR * hd; idx += BC) {
      int rr = idx / hd, e = idx % hd, i = qt * BR + rr;
      Qs[rr][e] = i < T ? q[base + (long)i * hd + e] : 0.f;
      dOs[rr][e] = i < T ? dout[base + (long)i * hd + e] : 0.f;
    }
    for (int rr = t; rr < BR; rr += BC) {
      int i = qt * BR + rr;
      Ls[rr] = i < T ? lse[(long)bh * T + i] : 0.f;
      Ds[rr] = i < T ? D[(long)bh * T + i] : 0.f;
    }
    __syncthreads();
    if (kr < T) {
      for (int rr = 0; rr < BR; rr++) {
        int i = qt * BR + rr;
        if (i >= T) break;
        if (i < kr) continue;              // causal: key kr only sees queries i>=kr
        float s = 0.f;
        for (int e = 0; e < hd; e++) s += Qs[rr][e] * Ks[t][e];
        float P = expf(s * scale - Ls[rr]);
        for (int e = 0; e < hd; e++) dVa[e] += P * dOs[rr][e];
        float dP = 0.f;
        for (int e = 0; e < hd; e++) dP += dOs[rr][e] * Vs[t][e];
        float dS = P * (dP - Ds[rr]);
        for (int e = 0; e < hd; e++) dKa[e] += scale * dS * Qs[rr][e];
      }
    }
    __syncthreads();
  }
  if (kr < T)
    for (int e = 0; e < hd; e++) {
      dk[base + (long)kr * hd + e] = dKa[e];
      dv[base + (long)kr * hd + e] = dVa[e];
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
  dim3 gq((T + BR - 1) / BR, B * H);
  flash_dq_k<<<gq, BR>>>(q, k, v, dout, lse, D, dq, T, hd, scale);
  dim3 gk((T + BC - 1) / BC, B * H);
  flash_dkv_k<<<gk, BC>>>(q, k, v, dout, lse, D, dk, dv, T, hd, scale);
  cudaFree(D);
}
