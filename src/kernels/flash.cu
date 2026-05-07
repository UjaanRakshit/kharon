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
