#ifndef KHARON_FLASH_H
#define KHARON_FLASH_H

// Hand-written FlashAttention (causal, FP32 accum). Tiled with online softmax;
// never materializes the [T,T] score matrix. q/k/v/o are [B,H,T,hd] contiguous.
// Forward also emits lse[B,H,T] (row log-sum-exp) for the backward pass.
#ifdef __cplusplus
extern "C" {
#endif

void flash_attn_fwd(const float *q, const float *k, const float *v,
                    float *o, float *lse, int B, int H, int T, int hd, float scale);
void flash_attn_fwd_bf(const void *q, const void *k, const void *v,
                       void *o, float *lse, int B, int H, int T, int hd, float scale);

void flash_attn_bwd(const float *q, const float *k, const float *v,
                    const float *o, const float *lse, const float *dout,
                    float *dq, float *dk, float *dv,
                    int B, int H, int T, int hd, float scale);
void flash_attn_bwd_bf(const void *q, const void *k, const void *v,
                       const void *o, const float *lse, const void *dout,
                       void *dq, void *dk, void *dv,
                       int B, int H, int T, int hd, float scale);

#ifdef __cplusplus
}
#endif

#endif
