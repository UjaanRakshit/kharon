#ifndef KHARON_KERNELS_H
#define KHARON_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

// cuBLAS handle lifecycle + row-major GEMM helpers. All matrices are row-major.
void gemm_init(void);
void gemm_destroy(void);
// C[M,N] = A[M,K] @ B[N,K]^T
void mm_nt(const float *A, const float *B, float *C, int M, int N, int K);
// BF16 tensor-core GEMM: A,B are bf16 (void*), C fp32, FP32 accumulation.
// C[M,N] = A[M,K] @ B[N,K]^T
void mm_nt_bf16(const void *A, const void *B, float *C, int M, int N, int K);
// dtype casts (bf16 buffers passed as void*)
void k_f2b(const float *in, void *out, long n);
void k_b2f(const void *in, float *out, long n);
// C[M,N] = A[M,K] @ B[K,N]
void mm_nn(const float *A, const float *B, float *C, int M, int N, int K);
// C[M,N] = A[K,M]^T @ B[K,N]
void mm_tn(const float *A, const float *B, float *C, int M, int N, int K);
// batched variants (constant strides), batch count = nb
void mm_nt_batched(const float *A, const float *B, float *C, int M, int N, int K,
                   long sA, long sB, long sC, int nb);
void mm_nn_batched(const float *A, const float *B, float *C, int M, int N, int K,
                   long sA, long sB, long sC, int nb);
// C[M,N] = A[K,M]^T @ B[K,N], batched
void mm_tn_batched(const float *A, const float *B, float *C, int M, int N, int K,
                   long sA, long sB, long sC, int nb);

// --- forward kernels ---
void k_embed(const float *wte, const float *wpe, const int *idx, float *out,
             int B, int T, int d);
void k_layernorm_fwd(const float *x, const float *w, const float *b,
                     float *out, float *mean, float *rstd, int rows, int d);
void k_add_bias(float *y, const float *b, int rows, int N);
void k_split_heads(const float *qkv, float *q, float *k, float *v,
                   int B, int T, int H, int hd);
void k_merge_heads(const float *atto, float *out, int B, int T, int H, int hd);
void k_softmax_causal_fwd(float *att, int rows_bh, int T, float scale);
void k_gelu_fwd(const float *x, float *y, long n);
void k_add(const float *a, const float *b, float *c, long n);
// fused: out = resid + (y + bias[col]); one pass instead of add_bias then add
void k_bias_residual(const float *y, const float *bias, const float *resid,
                     float *out, int rows, int N);
// fused: pre = y + bias[col]; act = gelu(pre); one pass instead of add_bias then gelu
void k_bias_gelu(const float *y, const float *bias, float *pre, float *act, int rows, int N);
void k_cross_entropy_fwd(const float *logits, const int *tgt, float *probs,
                         float *rowloss, int rows, int vocab);

// --- backward kernels ---
void k_cross_entropy_bwd(const float *probs, const int *tgt, float *dlogits,
                         int rows, int vocab, float invN);
void k_layernorm_bwd(const float *dy, const float *x, const float *w,
                     const float *mean, const float *rstd,
                     float *dx, float *dw, float *db, int rows, int d);
void k_gelu_bwd(const float *x, const float *dy, float *dx, long n);
void k_softmax_causal_bwd(const float *att, const float *datt, float *dscores,
                          int rows_bh, int T, float scale);
void k_unmerge_heads(const float *dout, float *datto, int B, int T, int H, int hd);
void k_combine_qkv(const float *dq, const float *dk, const float *dv, float *dqkv,
                   int B, int T, int H, int hd);
void k_colsum(const float *in, float *out, int rows, int N);   // out[j] += sum_rows in[:,j]
void k_embed_bwd_wte(const float *demb, const int *idx, float *dwte,
                     int rows, int vocab, int d);
void k_embed_bwd_wpe(const float *demb, float *dwpe, int B, int T, int d);

// --- optimizer ---
void k_adamw(float *p, const float *g, float *m, float *v, long n,
             float lr, float b1, float b2, float eps, float wd, float bc1, float bc2);

#ifdef __cplusplus
}
#endif

#endif
