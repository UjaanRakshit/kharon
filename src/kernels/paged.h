#ifndef KHARON_PAGED_H
#define KHARON_PAGED_H

// Inference (paged-KV decode) kernel launchers. fp32 path is the oracle; bf16 (_bf)
// is the throughput path. Cache layout per layer: [n_blocks, block_size, H, hd].

#ifdef __cplusplus
extern "C" {
#endif

void k_embed_pos(const float *wte, const float *wpe, const int *tok, const int *pos,
                 float *out, int d, long ntok);
void k_embed_pos_bf(const void *wte, const void *wpe, const int *tok, const int *pos,
                    void *out, int d, long ntok);
void k_append_kv(const float *qkv, float *kc, float *vc, const int *seqslot, const int *pos,
                 const int *btab, int max_lb, int bs, int H, int hd, int d, long ntok);
void k_append_kv_bf(const void *qkv, void *kc, void *vc, const int *seqslot, const int *pos,
                    const int *btab, int max_lb, int bs, int H, int hd, int d, long ntok);
void k_paged_attn(const float *qkv, const float *kc, const float *vc, const int *seqslot,
                  const int *pos, const int *btab, int max_lb, int bs, int H, int hd, int d,
                  float scale, float *out, long ntok);
void k_paged_attn_bf(const void *qkv, const void *kc, const void *vc, const int *seqslot,
                     const int *pos, const int *btab, int max_lb, int bs, int H, int hd, int d,
                     float scale, void *out, long ntok);
void k_gather_rows(const float *src, const int *rows, float *dst, int d, long nrow);
void k_gather_rows_bf(const void *src, const int *rows, void *dst, int d, long nrow);

#ifdef __cplusplus
}
#endif

#endif
