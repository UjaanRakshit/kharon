#include "model.h"
#include "kernels.h"
#include "kharon.h"
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// ---- layout helpers ----------------------------------------------------------
static void layout_weights(Arena *ar, Config c, Weights *w) {
  int d = c.d_model, V = c.vocab, S = c.seq, ff = 4 * d;
  w->wte = (float *)arena_alloc(ar, (long)V * d * 4);
  w->wpe = (float *)arena_alloc(ar, (long)S * d * 4);
  w->layer = (LayerW *)malloc(c.n_layer * sizeof(LayerW));
  long np = (long)V * d + (long)S * d;
  for (int l = 0; l < c.n_layer; l++) {
    LayerW *L = &w->layer[l];
    L->ln1_w = (float *)arena_alloc(ar, d * 4);     L->ln1_b = (float *)arena_alloc(ar, d * 4);
    L->qkv_w = (float *)arena_alloc(ar, (long)3 * d * d * 4); L->qkv_b = (float *)arena_alloc(ar, 3 * d * 4);
    L->proj_w = (float *)arena_alloc(ar, (long)d * d * 4);    L->proj_b = (float *)arena_alloc(ar, d * 4);
    L->ln2_w = (float *)arena_alloc(ar, d * 4);     L->ln2_b = (float *)arena_alloc(ar, d * 4);
    L->fc_w = (float *)arena_alloc(ar, (long)ff * d * 4);     L->fc_b = (float *)arena_alloc(ar, ff * 4);
    L->fcproj_w = (float *)arena_alloc(ar, (long)d * ff * 4); L->fcproj_b = (float *)arena_alloc(ar, d * 4);
    np += 4 * d + 3L * d * d + 3 * d + (long)d * d + d + (long)ff * d + ff + (long)d * ff + d;
  }
  w->lnf_w = (float *)arena_alloc(ar, d * 4);
  w->lnf_b = (float *)arena_alloc(ar, d * 4);
  np += 2 * d;
  w->cfg = c;
  w->n_param = (int)np;
}

static void layout_acts(Arena *ar, Config c, Acts *a) {
  int d = c.d_model, H = c.n_head, hd = d / H, V = c.vocab, ff = 4 * d;
  long R = (long)c.batch * c.seq, T = c.seq, BHT2 = (long)c.batch * H * T * T;
  a->emb = (float *)arena_alloc(ar, R * d * 4);
  a->layer = (LayerAct *)malloc(c.n_layer * sizeof(LayerAct));
  for (int l = 0; l < c.n_layer; l++) {
    LayerAct *A = &a->layer[l];
    A->ln1_mean = (float *)arena_alloc(ar, R * 4); A->ln1_rstd = (float *)arena_alloc(ar, R * 4);
    A->ln1 = (float *)arena_alloc(ar, R * d * 4);
    A->qkv = (float *)arena_alloc(ar, R * 3 * d * 4);
    A->q = (float *)arena_alloc(ar, R * d * 4); A->k = (float *)arena_alloc(ar, R * d * 4);
    A->v = (float *)arena_alloc(ar, R * d * 4);
    A->att = (float *)arena_alloc(ar, BHT2 * 4);
    A->atto = (float *)arena_alloc(ar, R * d * 4);
    A->atto_m = (float *)arena_alloc(ar, R * d * 4);
    A->proj = (float *)arena_alloc(ar, R * d * 4);
    A->res1 = (float *)arena_alloc(ar, R * d * 4);
    A->ln2_mean = (float *)arena_alloc(ar, R * 4); A->ln2_rstd = (float *)arena_alloc(ar, R * 4);
    A->ln2 = (float *)arena_alloc(ar, R * d * 4);
    A->fc = (float *)arena_alloc(ar, R * ff * 4);
    A->gelu = (float *)arena_alloc(ar, R * ff * 4);
    A->fcproj = (float *)arena_alloc(ar, R * d * 4);
    A->res2 = (float *)arena_alloc(ar, R * d * 4);
  }
  a->lnf_mean = (float *)arena_alloc(ar, R * 4); a->lnf_rstd = (float *)arena_alloc(ar, R * 4);
  a->lnf = (float *)arena_alloc(ar, R * d * 4);
  a->logits = (float *)arena_alloc(ar, R * V * 4);
  a->probs = (float *)arena_alloc(ar, R * V * 4);
  a->rowloss = (float *)arena_alloc(ar, R * 4);
}

static long weight_bytes(Config c) {
  int d = c.d_model, V = c.vocab, S = c.seq, ff = 4 * d;
  long per = 4L * d + 3L * d * d + 3 * d + (long)d * d + d + (long)ff * d + ff + (long)d * ff + d;
  return ((long)V * d + (long)S * d + c.n_layer * per + 2 * d) * 4;
}
static long act_bytes(Config c) {
  int d = c.d_model, H = c.n_head, V = c.vocab, ff = 4 * d;
  long R = (long)c.batch * c.seq, T = c.seq;
  long per = 2 * R + R * d + R * 3 * d + 3 * R * d + (long)c.batch * H * T * T
           + R * d + R * d + R * d + R * d + 2 * R + R * d + R * ff + R * ff + R * d + R * d;
  return (R * d + c.n_layer * per + 2 * R + R * d + R * V + R * V + R) * 4;
}

// ---- lifecycle ---------------------------------------------------------------
Model *model_create(Config cfg) {
  Model *m = (Model *)calloc(1, sizeof(Model));
  m->cfg = cfg;
  long wb = weight_bytes(cfg), ab = act_bytes(cfg);
  long slack = 1 << 18;
  m->w_arena = arena_create("params", wb + slack, 1);
  m->g_arena = arena_create("grads", wb + slack, 1);
  m->a_arena = arena_create("acts", ab + slack, 1);
  layout_weights(&m->w_arena, cfg, &m->w);
  layout_weights(&m->g_arena, cfg, &m->g);
  layout_acts(&m->a_arena, cfg, &m->a);
  long R = (long)cfg.batch * cfg.seq;
  CK(cudaMalloc((void **)&m->d_idx, R * 4));
  CK(cudaMalloc((void **)&m->d_tgt, R * 4));
  gemm_init();
  return m;
}

void model_free(Model *m) {
  if (!m) return;
  gemm_destroy();
  cudaFree(m->d_idx); cudaFree(m->d_tgt);
  arena_destroy(&m->w_arena); arena_destroy(&m->g_arena); arena_destroy(&m->a_arena);
  free(m->w.layer); free(m->g.layer); free(m->a.layer);
  free(m);
}

static void cpy(float *dst, RefFile *r, const char *name, long count) {
  RefTensor *t = ref_get(r, name);
  if (!t) DIE("model_load_ref: missing '%s'", name);
  if (t->count != count) DIE("model_load_ref: '%s' count %ld != %ld", name, t->count, count);
  CK(cudaMemcpy(dst, t->data, count * 4, cudaMemcpyHostToDevice));
}

void model_load_ref(Model *m, RefFile *r) {
  Config c = m->cfg;
  int d = c.d_model, V = c.vocab, S = c.seq, ff = 4 * d;
  cpy(m->w.wte, r, "wte", (long)V * d);
  cpy(m->w.wpe, r, "wpe", (long)S * d);
  char nm[64];
  for (int l = 0; l < c.n_layer; l++) {
    LayerW *L = &m->w.layer[l];
#define LD(field, suffix, cnt) do { snprintf(nm, sizeof(nm), "blk%d." suffix, l); cpy(L->field, r, nm, cnt); } while (0)
    LD(ln1_w, "ln1.w", d); LD(ln1_b, "ln1.b", d);
    LD(qkv_w, "qkv.w", (long)3 * d * d); LD(qkv_b, "qkv.b", 3 * d);
    LD(proj_w, "proj.w", (long)d * d); LD(proj_b, "proj.b", d);
    LD(ln2_w, "ln2.w", d); LD(ln2_b, "ln2.b", d);
    LD(fc_w, "fc.w", (long)ff * d); LD(fc_b, "fc.b", ff);
    LD(fcproj_w, "fcproj.w", (long)d * ff); LD(fcproj_b, "fcproj.b", d);
#undef LD
  }
  cpy(m->w.lnf_w, r, "lnf.w", d); cpy(m->w.lnf_b, r, "lnf.b", d);
}

void model_set_input(Model *m, const int *idx, const int *tgt) {
  long R = (long)m->cfg.batch * m->cfg.seq;
  CK(cudaMemcpy(m->d_idx, idx, R * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(m->d_tgt, tgt, R * 4, cudaMemcpyHostToDevice));
}

// ---- forward -----------------------------------------------------------------
float model_forward(Model *m) {
  Config c = m->cfg;
  int d = c.d_model, H = c.n_head, hd = d / H, V = c.vocab, ff = 4 * d, T = c.seq, B = c.batch;
  long R = (long)B * T;
  Weights *w = &m->w;
  Acts *a = &m->a;
  float scale = 1.f / sqrtf((float)hd);

  k_embed(w->wte, w->wpe, m->d_idx, a->emb, B, T, d);
  float *x = a->emb;
  for (int l = 0; l < c.n_layer; l++) {
    LayerW *L = &w->layer[l];
    LayerAct *A = &a->layer[l];
    k_layernorm_fwd(x, L->ln1_w, L->ln1_b, A->ln1, A->ln1_mean, A->ln1_rstd, R, d);
    mm_nt(A->ln1, L->qkv_w, A->qkv, R, 3 * d, d);
    k_add_bias(A->qkv, L->qkv_b, R, 3 * d);
    k_split_heads(A->qkv, A->q, A->k, A->v, B, T, H, hd);
    mm_nt_batched(A->q, A->k, A->att, T, T, hd, (long)T * hd, (long)T * hd, (long)T * T, B * H);
    k_softmax_causal_fwd(A->att, B * H * T, T, scale);
    mm_nn_batched(A->att, A->v, A->atto, T, hd, T, (long)T * T, (long)T * hd, (long)T * hd, B * H);
    k_merge_heads(A->atto, A->atto_m, B, T, H, hd);
    mm_nt(A->atto_m, L->proj_w, A->proj, R, d, d);
    k_add_bias(A->proj, L->proj_b, R, d);
    k_add(x, A->proj, A->res1, R * d);
    k_layernorm_fwd(A->res1, L->ln2_w, L->ln2_b, A->ln2, A->ln2_mean, A->ln2_rstd, R, d);
    mm_nt(A->ln2, L->fc_w, A->fc, R, ff, d);
    k_add_bias(A->fc, L->fc_b, R, ff);
    k_gelu_fwd(A->fc, A->gelu, R * ff);
    mm_nt(A->gelu, L->fcproj_w, A->fcproj, R, d, ff);
    k_add_bias(A->fcproj, L->fcproj_b, R, d);
    k_add(A->res1, A->fcproj, A->res2, R * d);
    x = A->res2;
  }
  k_layernorm_fwd(x, w->lnf_w, w->lnf_b, a->lnf, a->lnf_mean, a->lnf_rstd, R, d);
  mm_nt(a->lnf, w->wte, a->logits, R, V, d);
  k_cross_entropy_fwd(a->logits, m->d_tgt, a->probs, a->rowloss, R, V);

  float *h = (float *)malloc(R * 4);
  CK(cudaMemcpy(h, a->rowloss, R * 4, cudaMemcpyDeviceToHost));
  double s = 0;
  for (long i = 0; i < R; i++) s += h[i];
  free(h);
  m->loss = (float)(s / R);
  return m->loss;
}

void model_backward(Model *m) { (void)m; }  // filled in next
