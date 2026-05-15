#ifndef KHARON_MODEL_H
#define KHARON_MODEL_H

#include "arena.h"
#include "refio.h"
#include <stdint.h>

typedef struct {
  int n_layer, d_model, n_head, vocab, seq, batch;
} Config;

// Per-layer weights (and, with the same layout, per-layer grads).
typedef struct {
  float *ln1_w, *ln1_b;       // [d]
  float *qkv_w, *qkv_b;       // [3d,d], [3d]
  float *proj_w, *proj_b;     // [d,d], [d]
  float *ln2_w, *ln2_b;       // [d]
  float *fc_w, *fc_b;         // [4d,d], [4d]
  float *fcproj_w, *fcproj_b; // [d,4d], [d]
} LayerW;

typedef struct {
  Config cfg;
  float *wte;          // [vocab,d]   (head is tied to this)
  float *wpe;          // [seq,d]
  LayerW *layer;       // [n_layer]
  float *lnf_w, *lnf_b;// [d]
  int n_param;         // total scalar params
} Weights;

// Activations saved during forward for use in backward. Sized B*T rows.
typedef struct {
  float *ln1_mean, *ln1_rstd, *ln1;   // ln1: [BT,d]
  float *qkv;                         // [BT,3d]
  float *q, *k, *v;                   // [B,H,T,hd]
  float *att;                         // [B,H,T,T] (naive path; unused once FA is in)
  float *lse;                         // [B,H,T] FlashAttention log-sum-exp
  float *atto;                        // [B,H,T,hd] attention output
  float *atto_m;                      // [BT,d] heads merged
  float *proj;                        // [BT,d]
  float *res1;                        // [BT,d] x + attn
  float *ln2_mean, *ln2_rstd, *ln2;   // ln2: [BT,d]
  float *fc;                          // [BT,4d] pre-gelu
  float *gelu;                        // [BT,4d]
  float *fcproj;                      // [BT,d]
  float *res2;                        // [BT,d] block output
} LayerAct;

typedef struct {
  float *emb;                         // [BT,d] token+pos embedding
  LayerAct *layer;                    // [n_layer]
  float *lnf_mean, *lnf_rstd, *lnf;   // final LN: [BT,d]
  float *logits;                      // [BT,vocab]
  float *probs;                       // [BT,vocab] softmax
  float *rowloss;                     // [BT]
} Acts;

// Backward scratch (activation grads), reused across layers.
typedef struct {
  float *dx, *res1, *proj, *fcproj, *lntmp;   // [BT,d]
  float *ln1, *ln2;                           // [BT,d]
  float *fc, *gelu;                           // [BT,4d]
  float *atto_m, *atto;                       // [BT,d]
  float *att, *scores;                        // [B,H,T,T]
  float *q, *k, *v;                           // [BT,d]
  float *qkv;                                 // [BT,3d]
  float *lnf;                                 // [BT,d]
  float *logits;                              // [BT,vocab]
} Bwd;

typedef struct {
  Config cfg;
  Arena w_arena, g_arena, a_arena, s_arena;   // params, grads, activations, bwd scratch
  Arena wb_arena, ab_arena, sb_arena;         // bf16 weights + activations + bwd scratch
  Weights w, g, w_bf;
  Acts a, a_bf;
  Bwd s, s_bf;
  Arena om_arena, ov_arena;           // AdamW moment buffers (flat over params)
  float *opt_m, *opt_v;
  int step;
  int *d_idx, *d_tgt;                 // device input/target [BT]
  float loss;
} Model;

#ifdef __cplusplus
extern "C" {
#endif

Model *model_create(Config cfg);
void   model_free(Model *m);
void   model_load_ref(Model *m, RefFile *r);   // copy reference weights -> device
void   model_init_weights(Model *m, uint64_t seed);  // random GPT-2-style init
void   model_set_input(Model *m, const int *idx, const int *tgt);
float  model_forward(Model *m);                // returns loss
void   model_backward(Model *m);
void   model_adamw_step(Model *m, float lr, float b1, float b2, float eps, float wd);
void   model_sync_bf16(Model *m);              // cast fp32 master weights -> bf16 compute copy
float  model_forward_bf16(Model *m);           // bf16 mixed-precision forward (returns loss)
void   model_backward_bf16(Model *m);          // bf16 backward; weight grads accumulate fp32

#ifdef __cplusplus
}
#endif

#endif
