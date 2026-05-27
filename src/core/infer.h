#ifndef KHARON_INFER_H
#define KHARON_INFER_H

#include "model.h"

// Inference engine: paged KV-cache + continuous batching over the trained weights.
// fp32 path is the token-exact oracle (vs PyTorch greedy); bf16 is the throughput path.
// The cache is a pool of fixed-size blocks; per-sequence block tables map logical
// positions to physical blocks (vLLM-style), so no giant contiguous per-seq buffers.

typedef struct Engine Engine;

typedef struct {
  long blocks_total;      // pool size
  long blocks_peak;       // high-water physical blocks in use
  long tokens_decoded;    // generated (non-prompt) tokens
  long prefix_blocks_saved; // blocks saved by prefix sharing (group generate)
} InferStats;

#ifdef __cplusplus
extern "C" {
#endif

Engine *infer_create(Model *m, int block_size, int n_blocks, int max_tokens,
                     int max_seqs, int use_bf16);
void    infer_free(Engine *e);
// Sampling mode: temperature<=0 is greedy (deterministic, for the oracle); temperature>0
// draws categorically from softmax(logits/temperature) using a seeded RNG (for rollouts).
void    infer_set_sampling(Engine *e, float temperature, unsigned long long seed);

// Independent greedy generations via continuous batching (at most max_active in flight).
// out[i] (caller-allocated, >= plen[i]+n_new[i]) receives prompt+continuation; outlen[i]
// is set. Returns total tokens decoded; fills stats (nullable) and *ms wall time (nullable).
long infer_generate(Engine *e, int nseq, int **prompts, int *plen, int *n_new,
                    int max_active, int **out, int *outlen, InferStats *st, float *ms);

// G greedy samples from one shared prompt; the prompt KV is prefilled once and its blocks
// are shared (read-only) across all G sequences. Demonstrates prefix-sharing savings.
long infer_generate_group(Engine *e, int *prompt, int plen, int G, int n_new,
                          int **out, int *outlen, InferStats *st, float *ms);

#ifdef __cplusplus
}
#endif

#endif
