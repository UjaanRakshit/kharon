#include "model.h"
#include "kernels.h"
#include "flash.h"
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
    A->lse = (float *)arena_alloc(ar, (long)c.batch * H * T * 4);
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

static void layout_scratch(Arena *ar, Config c, Bwd *s) {
  int d = c.d_model, ff = 4 * d, V = c.vocab, H = c.n_head;
  long R = (long)c.batch * c.seq, T = c.seq, BHT2 = (long)c.batch * H * T * T;
  float **rd[] = {&s->dx, &s->res1, &s->proj, &s->fcproj, &s->lntmp, &s->ln1, &s->ln2,
                  &s->atto_m, &s->atto, &s->q, &s->k, &s->v, &s->lnf};
  for (int i = 0; i < (int)(sizeof(rd) / sizeof(rd[0])); i++)
    *rd[i] = (float *)arena_alloc(ar, R * d * 4);
  s->fc = (float *)arena_alloc(ar, R * ff * 4);
  s->gelu = (float *)arena_alloc(ar, R * ff * 4);
  s->att = (float *)arena_alloc(ar, BHT2 * 4);
  s->scores = (float *)arena_alloc(ar, BHT2 * 4);
  s->qkv = (float *)arena_alloc(ar, R * 3 * d * 4);
  s->logits = (float *)arena_alloc(ar, R * V * 4);
}

static long scratch_bytes(Config c) {
  int d = c.d_model, ff = 4 * d, V = c.vocab, H = c.n_head;
  long R = (long)c.batch * c.seq, T = c.seq;
  return (13 * R * d + 2 * R * ff + (long)c.batch * H * T * T * 2 + R * 3 * d + R * V) * 4;
}

static long weight_bytes(Config c) {
  int d = c.d_model, V = c.vocab, S = c.seq, ff = 4 * d;
  long per = 4L * d + 3L * d * d + 3 * d + (long)d * d + d + (long)ff * d + ff + (long)d * ff + d;
  return ((long)V * d + (long)S * d + c.n_layer * per + 2 * d) * 4;
}
static long act_bytes(Config c) {
  int d = c.d_model, H = c.n_head, V = c.vocab, ff = 4 * d;
  long R = (long)c.batch * c.seq, T = c.seq;
  long per = 2 * R + R * d + R * 3 * d + 3 * R * d + (long)c.batch * H * T * T + (long)c.batch * H * T
           + R * d + R * d + R * d + R * d + 2 * R + R * d + R * ff + R * ff + R * d + R * d;
  return (R * d + c.n_layer * per + 2 * R + R * d + R * V + R * V + R) * 4;
}

// ---- lifecycle ---------------------------------------------------------------
Model *model_create(Config cfg) {
  Model *m = (Model *)calloc(1, sizeof(Model));
  m->cfg = cfg;
  long wb = weight_bytes(cfg), ab = act_bytes(cfg);
  long slack = 1 << 18;
  long sb = scratch_bytes(cfg);
  m->w_arena = arena_create("params", wb + slack, 1);
  m->g_arena = arena_create("grads", wb + slack, 1);
  m->a_arena = arena_create("acts", ab + slack, 1);
  m->s_arena = arena_create("bwd", sb + slack, 1);
  layout_weights(&m->w_arena, cfg, &m->w);
  layout_weights(&m->g_arena, cfg, &m->g);
  layout_acts(&m->a_arena, cfg, &m->a);
  layout_scratch(&m->s_arena, cfg, &m->s);
  // optimizer moments are flat over the whole param arena (identical layout)
  long pbytes = m->w_arena.off;
  m->om_arena = arena_create("adam_m", pbytes, 1);
  m->ov_arena = arena_create("adam_v", pbytes, 1);
  m->opt_m = (float *)arena_alloc(&m->om_arena, pbytes);
  m->opt_v = (float *)arena_alloc(&m->ov_arena, pbytes);
  CK(cudaMemset(m->opt_m, 0, pbytes));
  CK(cudaMemset(m->opt_v, 0, pbytes));
  m->step = 0;
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
  arena_destroy(&m->w_arena); arena_destroy(&m->g_arena);
  arena_destroy(&m->a_arena); arena_destroy(&m->s_arena);
  arena_destroy(&m->om_arena); arena_destroy(&m->ov_arena);
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
    flash_attn_fwd(A->q, A->k, A->v, A->atto, A->lse, B, H, T, hd, scale);
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

static void dcopy(float *dst, const float *src, long n) {
  CK(cudaMemcpy(dst, src, n * 4, cudaMemcpyDeviceToDevice));
}

void model_backward(Model *m) {
  Config c = m->cfg;
  int d = c.d_model, H = c.n_head, hd = d / H, V = c.vocab, ff = 4 * d, T = c.seq, B = c.batch;
  long R = (long)B * T, RD = R * d;
  float scale = 1.f / sqrtf((float)hd);
  Weights *w = &m->w, *g = &m->g;
  Acts *a = &m->a;
  Bwd *s = &m->s;

  CK(cudaMemset(m->g_arena.base, 0, m->g_arena.off));   // zero all grads

  // cross entropy + head (logits = lnf @ wte^T)
  k_cross_entropy_bwd(a->probs, m->d_tgt, s->logits, R, V, 1.f / R);
  mm_nn(s->logits, w->wte, s->lnf, R, d, V);
  mm_tn(s->logits, a->lnf, g->wte, V, d, R);

  float *xlast = c.n_layer ? a->layer[c.n_layer - 1].res2 : a->emb;
  k_layernorm_bwd(s->lnf, xlast, w->lnf_w, a->lnf_mean, a->lnf_rstd,
                  s->dx, g->lnf_w, g->lnf_b, R, d);

  for (int l = c.n_layer - 1; l >= 0; l--) {
    LayerW *L = &w->layer[l];
    LayerW *gL = &g->layer[l];
    LayerAct *A = &a->layer[l];
    float *xin = l ? a->layer[l - 1].res2 : a->emb;

    // res2 = res1 + fcproj : dx flows to both
    dcopy(s->res1, s->dx, RD);
    dcopy(s->fcproj, s->dx, RD);
    // fcproj = gelu @ fcproj_w^T + b
    k_colsum(s->fcproj, gL->fcproj_b, R, d);
    mm_tn(s->fcproj, A->gelu, gL->fcproj_w, d, ff, R);
    mm_nn(s->fcproj, L->fcproj_w, s->gelu, R, ff, d);
    // gelu
    k_gelu_bwd(A->fc, s->gelu, s->fc, R * ff);
    // fc = ln2 @ fc_w^T + b
    k_colsum(s->fc, gL->fc_b, R, ff);
    mm_tn(s->fc, A->ln2, gL->fc_w, ff, d, R);
    mm_nn(s->fc, L->fc_w, s->ln2, R, d, ff);
    // ln2 = LN(res1) : accumulate into res1
    k_layernorm_bwd(s->ln2, A->res1, L->ln2_w, A->ln2_mean, A->ln2_rstd,
                    s->lntmp, gL->ln2_w, gL->ln2_b, R, d);
    k_add(s->res1, s->lntmp, s->res1, RD);
    // res1 = x + proj : dx(base) = res1, proj branch = res1
    dcopy(s->dx, s->res1, RD);
    dcopy(s->proj, s->res1, RD);
    // proj = atto_m @ proj_w^T + b
    k_colsum(s->proj, gL->proj_b, R, d);
    mm_tn(s->proj, A->atto_m, gL->proj_w, d, d, R);
    mm_nn(s->proj, L->proj_w, s->atto_m, R, d, d);
    // atto_m = merge(atto)
    k_unmerge_heads(s->atto_m, s->atto, B, T, H, hd);   // s->atto = grad of attn output
    flash_attn_bwd(A->q, A->k, A->v, A->atto, A->lse, s->atto, s->q, s->k, s->v, B, H, T, hd, scale);
    k_combine_qkv(s->q, s->k, s->v, s->qkv, B, T, H, hd);
    // qkv = ln1 @ qkv_w^T + b
    k_colsum(s->qkv, gL->qkv_b, R, 3 * d);
    mm_tn(s->qkv, A->ln1, gL->qkv_w, 3 * d, d, R);
    mm_nn(s->qkv, L->qkv_w, s->ln1, R, d, 3 * d);
    // ln1 = LN(xin) : accumulate into dx
    k_layernorm_bwd(s->ln1, xin, L->ln1_w, A->ln1_mean, A->ln1_rstd,
                    s->lntmp, gL->ln1_w, gL->ln1_b, R, d);
    k_add(s->dx, s->lntmp, s->dx, RD);
  }

  // embedding: dx is grad of emb
  k_embed_bwd_wte(s->dx, m->d_idx, g->wte, R, V, d);
  k_embed_bwd_wpe(s->dx, g->wpe, B, T, d);
}
