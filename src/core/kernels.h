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
// C[M,N] = A[M,K] @ B[K,N]
void mm_nn(const float *A, const float *B, float *C, int M, int N, int K);
// C[M,N] = A[K,M]^T @ B[K,N]
void mm_tn(const float *A, const float *B, float *C, int M, int N, int K);
// batched variants (constant strides), batch count = nb
void mm_nt_batched(const float *A, const float *B, float *C, int M, int N, int K,
                   long sA, long sB, long sC, int nb);
void mm_nn_batched(const float *A, const float *B, float *C, int M, int N, int K,
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
void k_cross_entropy_fwd(const float *logits, const int *tgt, float *probs,
                         float *rowloss, int rows, int vocab);

#ifdef __cplusplus
}
#endif

#endif
