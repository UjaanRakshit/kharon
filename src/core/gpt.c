#include "model.h"
#include "kernels.h"
#include "flash.h"
#include "rng.h"
#include "kharon.h"
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// ---- layout helpers ----------------------------------------------------------
// es = element size in bytes (4 for fp32, 2 for bf16). Pointers are stored as
// float* but for bf16 they hold bf16 addresses (only ever passed to _bf launchers
// as void*). Reduction stats (mean/rstd/lse/probs/rowloss) are always read from
// the fp32 acts, so their bf16-arena copies are unused.
// tp = tensor-parallel size. Column-parallel (qkv,fc) shard output; row-parallel
// (proj,fcproj) shard input. Embeddings, LayerNorms, proj/fcproj bias stay full.
// tp=1 -> identical to the single-GPU layout.
static void layout_weights(Arena *ar, Config c, Weights *w, int es, int tp, int has_head) {
  int d = c.d_model, V = c.vocab, S = c.seq, ff = 4 * d;
  int qo = 3 * d / tp, pi = d / tp, fo = ff / tp, fpi = ff / tp;
  w->wte = (float *)arena_alloc(ar, (long)V * d * es);
  w->wpe = (float *)arena_alloc(ar, (long)S * d * es);
  w->layer = (LayerW *)malloc(c.n_layer * sizeof(LayerW));
  long np = (long)V * d + (long)S * d;
  for (int l = 0; l < c.n_layer; l++) {
    LayerW *L = &w->layer[l];
    L->ln1_w = (float *)arena_alloc(ar, d * es);     L->ln1_b = (float *)arena_alloc(ar, d * es);
    L->qkv_w = (float *)arena_alloc(ar, (long)qo * d * es); L->qkv_b = (float *)arena_alloc(ar, qo * es);
    L->proj_w = (float *)arena_alloc(ar, (long)d * pi * es); L->proj_b = (float *)arena_alloc(ar, d * es);
    L->ln2_w = (float *)arena_alloc(ar, d * es);     L->ln2_b = (float *)arena_alloc(ar, d * es);
    L->fc_w = (float *)arena_alloc(ar, (long)fo * d * es);  L->fc_b = (float *)arena_alloc(ar, fo * es);
    L->fcproj_w = (float *)arena_alloc(ar, (long)d * fpi * es); L->fcproj_b = (float *)arena_alloc(ar, d * es);
    np += 4 * d + (long)qo * d + qo + (long)d * pi + d + (long)fo * d + fo + (long)d * fpi + d;
  }
  w->lnf_w = (float *)arena_alloc(ar, d * es);
  w->lnf_b = (float *)arena_alloc(ar, d * es);
  np += 2 * d;
  if (has_head) { w->head_w = (float *)arena_alloc(ar, (long)V * d * es); np += (long)V * d; }
  else w->head_w = NULL;
  w->cfg = c;
  w->n_param = (int)np;
}

static void layout_acts(Arena *ar, Config c, Acts *a, int es, int tp) {
  int d = c.d_model, H = c.n_head, V = c.vocab, ff = 4 * d;
  int dp = d / tp, qo = 3 * d / tp, fo = ff / tp, Hp = H / tp;
  long R = (long)c.batch * c.seq, T = c.seq, BHT2 = (long)c.batch * Hp * T * T;
  a->emb = (float *)arena_alloc(ar, R * d * es);
  a->layer = (LayerAct *)malloc(c.n_layer * sizeof(LayerAct));
  for (int l = 0; l < c.n_layer; l++) {
    LayerAct *A = &a->layer[l];
    A->ln1_mean = (float *)arena_alloc(ar, R * 4); A->ln1_rstd = (float *)arena_alloc(ar, R * 4);
    A->ln1 = (float *)arena_alloc(ar, R * d * es);              // replicated
    A->qkv = (float *)arena_alloc(ar, R * qo * es);             // sharded (col)
    A->q = (float *)arena_alloc(ar, R * dp * es); A->k = (float *)arena_alloc(ar, R * dp * es);
    A->v = (float *)arena_alloc(ar, R * dp * es);
    A->att = (float *)arena_alloc(ar, BHT2 * es);
    A->lse = (float *)arena_alloc(ar, (long)c.batch * Hp * T * 4);
    A->atto = (float *)arena_alloc(ar, R * dp * es);
    A->atto_m = (float *)arena_alloc(ar, R * dp * es);
    A->proj = (float *)arena_alloc(ar, R * d * es);             // replicated (row out)
    A->res1 = (float *)arena_alloc(ar, R * d * es);
    A->ln2_mean = (float *)arena_alloc(ar, R * 4); A->ln2_rstd = (float *)arena_alloc(ar, R * 4);
    A->ln2 = (float *)arena_alloc(ar, R * d * es);
    A->fc = (float *)arena_alloc(ar, R * fo * es);              // sharded (col)
    A->gelu = (float *)arena_alloc(ar, R * fo * es);
    A->fcproj = (float *)arena_alloc(ar, R * d * es);           // replicated (row out)
    A->res2 = (float *)arena_alloc(ar, R * d * es);
  }
  a->lnf_mean = (float *)arena_alloc(ar, R * 4); a->lnf_rstd = (float *)arena_alloc(ar, R * 4);
  a->lnf = (float *)arena_alloc(ar, R * d * es);
  a->logits = (float *)arena_alloc(ar, R * V * es);
  a->probs = (float *)arena_alloc(ar, R * V * 4);
  a->rowloss = (float *)arena_alloc(ar, R * 4);
}

static void layout_scratch(Arena *ar, Config c, Bwd *s, int es, int tp) {
  int d = c.d_model, ff = 4 * d, V = c.vocab, H = c.n_head;
  int dp = d / tp, qo = 3 * d / tp, fo = ff / tp, Hp = H / tp;
  long R = (long)c.batch * c.seq, T = c.seq, BHT2 = (long)c.batch * Hp * T * T;
  float **full[] = {&s->dx, &s->res1, &s->proj, &s->fcproj, &s->lntmp, &s->ln1, &s->ln2, &s->lnf};
  for (int i = 0; i < (int)(sizeof(full) / sizeof(full[0])); i++)
    *full[i] = (float *)arena_alloc(ar, R * d * es);
  float **shard[] = {&s->atto_m, &s->atto, &s->q, &s->k, &s->v};
  for (int i = 0; i < (int)(sizeof(shard) / sizeof(shard[0])); i++)
    *shard[i] = (float *)arena_alloc(ar, R * dp * es);
  s->fc = (float *)arena_alloc(ar, R * fo * es);
  s->gelu = (float *)arena_alloc(ar, R * fo * es);
  s->att = (float *)arena_alloc(ar, BHT2 * es);
  s->scores = (float *)arena_alloc(ar, BHT2 * es);
  s->qkv = (float *)arena_alloc(ar, R * qo * es);
  s->logits = (float *)arena_alloc(ar, R * V * es);
}

static long scratch_bytes(Config c, int tp) {
  int d = c.d_model, ff = 4 * d, V = c.vocab, H = c.n_head;
  int dp = d / tp, qo = 3 * d / tp, fo = ff / tp, Hp = H / tp;
  long R = (long)c.batch * c.seq, T = c.seq;
  return (8 * R * d + 5 * R * dp + 2 * R * fo + (long)c.batch * Hp * T * T * 2 + R * qo + R * V) * 4;
}

static long weight_bytes(Config c, int tp, int has_head) {
  int d = c.d_model, V = c.vocab, S = c.seq, ff = 4 * d;
  int qo = 3 * d / tp, pi = d / tp, fo = ff / tp, fpi = ff / tp;
  long per = 4L * d + (long)qo * d + qo + (long)d * pi + d + (long)fo * d + fo + (long)d * fpi + d;
  long tot = (long)V * d + (long)S * d + c.n_layer * per + 2 * d + (has_head ? (long)V * d : 0);
  return tot * 4;
}
static long act_bytes(Config c, int tp) {
  int d = c.d_model, H = c.n_head, V = c.vocab, ff = 4 * d;
  int dp = d / tp, qo = 3 * d / tp, fo = ff / tp, Hp = H / tp;
  long R = (long)c.batch * c.seq, T = c.seq;
  long per = 2 * R + R * d + R * qo + 3 * R * dp + (long)c.batch * Hp * T * T + (long)c.batch * Hp * T
           + R * dp + R * dp + R * d + R * d + 2 * R + R * d + R * fo + R * fo + R * d + R * d;
  return (R * d + c.n_layer * per + 2 * R + R * d + R * V + R * V + R) * 4;
}

// ---- lifecycle ---------------------------------------------------------------
static Model *model_alloc(Config cfg, int tp, int rank, int has_head) {
  Model *m = (Model *)calloc(1, sizeof(Model));
  m->cfg = cfg;
  m->tp = tp;
  m->rank = rank;
  long wb = weight_bytes(cfg, tp, has_head), ab = act_bytes(cfg, tp);
  long slack = 1 << 18;
  long sb = scratch_bytes(cfg, tp);
  m->w_arena = arena_create("params", wb + slack, 1);
  m->g_arena = arena_create("grads", wb + slack, 1);
  m->a_arena = arena_create("acts", ab + slack, 1);
  m->s_arena = arena_create("bwd", sb + slack, 1);
  layout_weights(&m->w_arena, cfg, &m->w, 4, tp, has_head);
  layout_weights(&m->g_arena, cfg, &m->g, 4, tp, has_head);
  layout_acts(&m->a_arena, cfg, &m->a, 4, tp);
  layout_scratch(&m->s_arena, cfg, &m->s, 4, tp);
  m->wb_arena = arena_create("wbf16", wb / 2 + slack, 1);
  m->ab_arena = arena_create("abf16", ab + slack, 1);   // bf16 acts but fp32 stats don't halve
  m->sb_arena = arena_create("sbf16", sb / 2 + slack, 1);
  layout_weights(&m->wb_arena, cfg, &m->w_bf, 2, tp, has_head);
  layout_acts(&m->ab_arena, cfg, &m->a_bf, 2, tp);
  layout_scratch(&m->sb_arena, cfg, &m->s_bf, 2, tp);
  long pbytes = m->w_arena.off;                          // optimizer flat over params
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

Model *model_create(Config cfg) { return model_alloc(cfg, 1, 0, 0); }
Model *model_create_tp(Config cfg, int tp, int rank) { return model_alloc(cfg, tp, rank, 0); }

// Pipeline stage: stage_cfg.n_layer = this stage's layer count. Last stage gets the
// untied head; non-last stages allocate xin/dxout staging buffers for the scheduler.
Model *model_create_pp(Config stage_cfg, int first, int last, int nslots, int tp, int tp_rank) {
  Model *m = model_alloc(stage_cfg, tp, tp_rank, last);
  m->pp_first = first;
  m->pp_last = last;
  long bt_d = (long)stage_cfg.batch * stage_cfg.seq * stage_cfg.d_model * 2;  // bf16 [B,T,d]
  if (!first) CK(cudaMalloc(&m->pp_xin, bt_d));     // recv activation here
  if (!last) CK(cudaMalloc(&m->pp_dxout, bt_d));    // recv grad-of-output here
  if (nslots > 8) DIE("model_create_pp: nslots %d > 8", nslots);
  m->pp_nslots = nslots;
  m->cur_slot = 0;
  long ab = act_bytes(stage_cfg, tp), slack = 1 << 18;
  m->a_slot[0] = m->a; m->a_bf_slot[0] = m->a_bf;   // slot 0 aliases base acts
  for (int s = 1; s < nslots; s++) {
    m->a_slot_ar[s] = arena_create("acts_s", ab + slack, 1);
    m->ab_slot_ar[s] = arena_create("abf_s", ab + slack, 1);
    layout_acts(&m->a_slot_ar[s], stage_cfg, &m->a_slot[s], 4, tp);
    layout_acts(&m->ab_slot_ar[s], stage_cfg, &m->a_bf_slot[s], 2, tp);
  }
  return m;
}
void model_set_slot(Model *m, int slot) { m->cur_slot = slot; }

void model_free(Model *m) {
  if (!m) return;
  gemm_destroy();
  cudaFree(m->d_idx); cudaFree(m->d_tgt);
  arena_destroy(&m->w_arena); arena_destroy(&m->g_arena);
  arena_destroy(&m->a_arena); arena_destroy(&m->s_arena);
  arena_destroy(&m->om_arena); arena_destroy(&m->ov_arena);
  arena_destroy(&m->wb_arena); arena_destroy(&m->ab_arena); arena_destroy(&m->sb_arena);
  cudaFree(m->pp_xin); cudaFree(m->pp_dxout);
  for (int s = 1; s < m->pp_nslots; s++) {
    arena_destroy(&m->a_slot_ar[s]); arena_destroy(&m->ab_slot_ar[s]);
    free(m->a_slot[s].layer); free(m->a_bf_slot[s].layer);
  }
  free(m->w.layer); free(m->g.layer); free(m->a.layer);
  free(m->w_bf.layer); free(m->a_bf.layer);
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

static float rnd_normal(Rng *st) {           // Box-Muller, one sample
  float u1 = (rng_u32(st) + 1.f) / 4294967296.f;
  float u2 = (float)rng_u32(st) / 4294967296.f;
  return sqrtf(-2.f * logf(u1)) * cosf(6.2831853f * u2);
}
static void fill_normal(float *dptr, long n, float std, Rng *st) {
  float *h = (float *)malloc(n * 4);
  for (long i = 0; i < n; i++) h[i] = std * rnd_normal(st);
  CK(cudaMemcpy(dptr, h, n * 4, cudaMemcpyHostToDevice));
  free(h);
}
static void fill_const(float *dptr, long n, float v) {
  float *h = (float *)malloc(n * 4);
  for (long i = 0; i < n; i++) h[i] = v;
  CK(cudaMemcpy(dptr, h, n * 4, cudaMemcpyHostToDevice));
  free(h);
}

// GPT-2-style init: matmuls + embeddings ~ N(0,0.02), biases 0, LayerNorm w=1 b=0.
void model_init_weights(Model *m, uint64_t seed) {
  Config c = m->cfg;
  int d = c.d_model, V = c.vocab, S = c.seq, ff = 4 * d;
  Rng st; rng_seed(&st, seed);
  Weights *w = &m->w;
  fill_normal(w->wte, (long)V * d, 0.02f, &st);
  fill_normal(w->wpe, (long)S * d, 0.02f, &st);
  for (int l = 0; l < c.n_layer; l++) {
    LayerW *L = &w->layer[l];
    fill_const(L->ln1_w, d, 1.f); fill_const(L->ln1_b, d, 0.f);
    fill_normal(L->qkv_w, (long)3 * d * d, 0.02f, &st); fill_const(L->qkv_b, 3 * d, 0.f);
    fill_normal(L->proj_w, (long)d * d, 0.02f, &st); fill_const(L->proj_b, d, 0.f);
    fill_const(L->ln2_w, d, 1.f); fill_const(L->ln2_b, d, 0.f);
    fill_normal(L->fc_w, (long)ff * d, 0.02f, &st); fill_const(L->fc_b, ff, 0.f);
    fill_normal(L->fcproj_w, (long)d * ff, 0.02f, &st); fill_const(L->fcproj_b, d, 0.f);
  }
  fill_const(w->lnf_w, d, 1.f); fill_const(w->lnf_b, d, 0.f);
}

// Generate the FULL [Rf,Cf] normal matrix (same RNG draw as the single-GPU init),
// then copy the [Rs,Cs] block at (r0,c0) to the device shard. Identical RNG order
// across ranks => the shards reassemble to the same weights as single-GPU.
// NULL dptr => consume RNG only (keeps the global weight stream in sync across ranks
// that don't own this tensor — needed for PP + PP*TP consistency).
static void fill_normal_block(float *dptr, int Rf, int Cf, int r0, int Rs, int c0, int Cs,
                              float std, Rng *st) {
  float *full = (float *)malloc((long)Rf * Cf * 4);
  for (long i = 0; i < (long)Rf * Cf; i++) full[i] = std * rnd_normal(st);
  if (dptr) {
    float *sh = (float *)malloc((long)Rs * Cs * 4);
    for (int i = 0; i < Rs; i++)
      for (int j = 0; j < Cs; j++) sh[(long)i * Cs + j] = full[(long)(r0 + i) * Cf + (c0 + j)];
    CK(cudaMemcpy(dptr, sh, (long)Rs * Cs * 4, cudaMemcpyHostToDevice));
    free(sh);
  }
  free(full);
}
// qkv: full [3d,d] (q,k,v stacked); rank r takes dp rows from each of q,k,v.
static void init_qkv_shard(float *dptr, int d, int tp, int r, Rng *st) {
  int dp = d / tp;
  float *full = (float *)malloc((long)3 * d * d * 4);
  for (long i = 0; i < 3L * d * d; i++) full[i] = 0.02f * rnd_normal(st);
  if (dptr) {
    float *sh = (float *)malloc((long)3 * dp * d * 4);
    for (int part = 0; part < 3; part++)
      for (int i = 0; i < dp; i++)
        for (int j = 0; j < d; j++)
          sh[((long)part * dp + i) * d + j] = full[((long)part * d + r * dp + i) * d + j];
    CK(cudaMemcpy(dptr, sh, (long)3 * dp * d * 4, cudaMemcpyHostToDevice));
    free(sh);
  }
  free(full);
}

void model_init_weights_tp(Model *m, uint64_t seed) {
  Config c = m->cfg;
  int d = c.d_model, V = c.vocab, S = c.seq, ff = 4 * d;
  int tp = m->tp, r = m->rank, dp = d / tp, fo = ff / tp;
  Rng st; rng_seed(&st, seed);
  Weights *w = &m->w;
  fill_normal(w->wte, (long)V * d, 0.02f, &st);   // replicated; same draw as single-GPU
  fill_normal(w->wpe, (long)S * d, 0.02f, &st);
  for (int l = 0; l < c.n_layer; l++) {
    LayerW *L = &w->layer[l];
    fill_const(L->ln1_w, d, 1.f); fill_const(L->ln1_b, d, 0.f);
    init_qkv_shard(L->qkv_w, d, tp, r, &st);             // column-parallel
    fill_const(L->qkv_b, 3 * dp, 0.f);
    fill_normal_block(L->proj_w, d, d, 0, d, r * dp, dp, 0.02f, &st);   // row-parallel (cols)
    fill_const(L->proj_b, d, 0.f);
    fill_const(L->ln2_w, d, 1.f); fill_const(L->ln2_b, d, 0.f);
    fill_normal_block(L->fc_w, ff, d, r * fo, fo, 0, d, 0.02f, &st);    // column-parallel (rows)
    fill_const(L->fc_b, fo, 0.f);
    fill_normal_block(L->fcproj_w, d, ff, 0, d, r * fo, fo, 0.02f, &st);// row-parallel (cols)
    fill_const(L->fcproj_b, d, 0.f);
  }
  fill_const(w->lnf_w, d, 1.f); fill_const(w->lnf_b, d, 0.f);
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
    k_bias_residual(A->proj, L->proj_b, x, A->res1, R, d);
    k_layernorm_fwd(A->res1, L->ln2_w, L->ln2_b, A->ln2, A->ln2_mean, A->ln2_rstd, R, d);
    mm_nt(A->ln2, L->fc_w, A->fc, R, ff, d);
    k_bias_gelu(A->fc, L->fc_b, A->fc, A->gelu, R, ff);
    mm_nt(A->gelu, L->fcproj_w, A->fcproj, R, d, ff);
    k_bias_residual(A->fcproj, L->fcproj_b, A->res1, A->res2, R, d);
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

void model_sync_bf16(Model *m) {
  Config c = m->cfg;
  int d = c.d_model, V = c.vocab, S = c.seq, ff = 4 * d, tp = m->tp;
  long qo = 3L * d / tp, pi = (long)d / tp, fo = (long)ff / tp;   // sharded counts
  Weights *w = &m->w, *b = &m->w_bf;
  k_f2b(w->wte, b->wte, (long)V * d);
  k_f2b(w->wpe, b->wpe, (long)S * d);
  for (int l = 0; l < c.n_layer; l++) {
    LayerW *L = &w->layer[l], *B = &b->layer[l];
    k_f2b(L->ln1_w, B->ln1_w, d); k_f2b(L->ln1_b, B->ln1_b, d);
    k_f2b(L->qkv_w, B->qkv_w, qo * d); k_f2b(L->qkv_b, B->qkv_b, qo);
    k_f2b(L->proj_w, B->proj_w, (long)d * pi); k_f2b(L->proj_b, B->proj_b, d);
    k_f2b(L->ln2_w, B->ln2_w, d); k_f2b(L->ln2_b, B->ln2_b, d);
    k_f2b(L->fc_w, B->fc_w, fo * d); k_f2b(L->fc_b, B->fc_b, fo);
    k_f2b(L->fcproj_w, B->fcproj_w, (long)d * fo); k_f2b(L->fcproj_b, B->fcproj_b, d);
  }
  k_f2b(w->lnf_w, b->lnf_w, d); k_f2b(w->lnf_b, b->lnf_b, d);
  if (w->head_w) k_f2b(w->head_w, b->head_w, (long)V * d);
}

// BF16 mixed-precision forward: bf16 activations/weights through tensor-core GEMMs,
// fp32 reductions (stats live in the fp32 acts arena). Master weights stay fp32.
float model_forward_bf16(Model *m) {
  Config c = m->cfg;
  int d = c.d_model, H = c.n_head, hd = d / H, V = c.vocab, ff = 4 * d, T = c.seq, B = c.batch;
  long R = (long)B * T;
  Weights *w = &m->w_bf;
  Acts *a = &m->a_bf, *af = &m->a;
  float scale = 1.f / sqrtf((float)hd);

  k_embed_bf(w->wte, w->wpe, m->d_idx, a->emb, B, T, d);
  void *x = a->emb;
  for (int l = 0; l < c.n_layer; l++) {
    LayerW *W = &w->layer[l];
    LayerAct *A = &a->layer[l], *F = &af->layer[l];
    k_layernorm_fwd_bf(x, W->ln1_w, W->ln1_b, A->ln1, F->ln1_mean, F->ln1_rstd, R, d);
    mm_nt_bf16o(A->ln1, W->qkv_w, A->qkv, R, 3 * d, d);
    k_add_bias_bf(A->qkv, W->qkv_b, R, 3 * d);
    k_split_heads_bf(A->qkv, A->q, A->k, A->v, B, T, H, hd);
    flash_attn_fwd_bf(A->q, A->k, A->v, A->atto, F->lse, B, H, T, hd, scale);
    k_merge_heads_bf(A->atto, A->atto_m, B, T, H, hd);
    mm_nt_bf16o(A->atto_m, W->proj_w, A->proj, R, d, d);
    k_bias_residual_bf(A->proj, W->proj_b, x, A->res1, R, d);
    k_layernorm_fwd_bf(A->res1, W->ln2_w, W->ln2_b, A->ln2, F->ln2_mean, F->ln2_rstd, R, d);
    mm_nt_bf16o(A->ln2, W->fc_w, A->fc, R, ff, d);
    k_bias_gelu_bf(A->fc, W->fc_b, A->fc, A->gelu, R, ff);
    mm_nt_bf16o(A->gelu, W->fcproj_w, A->fcproj, R, d, ff);
    k_bias_residual_bf(A->fcproj, W->fcproj_b, A->res1, A->res2, R, d);
    x = A->res2;
  }
  k_layernorm_fwd_bf(x, w->lnf_w, w->lnf_b, a->lnf, af->lnf_mean, af->lnf_rstd, R, d);
  mm_nt_bf16o(a->lnf, w->wte, a->logits, R, V, d);
  k_cross_entropy_fwd_bf(a->logits, m->d_tgt, af->probs, af->rowloss, R, V);

  float *h = (float *)malloc(R * 4);
  CK(cudaMemcpy(h, af->rowloss, R * 4, cudaMemcpyDeviceToHost));
  double s = 0;
  for (long i = 0; i < R; i++) s += h[i];
  free(h);
  m->loss = (float)(s / R);
  return m->loss;
}

// Tensor-parallel bf16 forward. Column-parallel qkv/fc (output sharded, no comm);
// row-parallel proj/fcproj (input sharded, output all-reduced). Embeddings, LNs,
// head replicated. tp=1 with allreduce==NULL is identical to model_forward_bf16.
float model_forward_tp(Model *m) {
  Config c = m->cfg;
  int d = c.d_model, H = c.n_head, hd = d / H, V = c.vocab, ff = 4 * d, T = c.seq, B = c.batch;
  int tp = m->tp, dp = d / tp, Hp = H / tp, qo = 3 * d / tp, fo = ff / tp;
  long R = (long)B * T;
  float scale = 1.f / sqrtf((float)hd);
  Weights *w = &m->w_bf;
  Acts *a = &m->a_bf, *af = &m->a;

  k_embed_bf(w->wte, w->wpe, m->d_idx, a->emb, B, T, d);
  void *x = a->emb;
  for (int l = 0; l < c.n_layer; l++) {
    LayerW *W = &w->layer[l];
    LayerAct *A = &a->layer[l], *F = &af->layer[l];
    k_layernorm_fwd_bf(x, W->ln1_w, W->ln1_b, A->ln1, F->ln1_mean, F->ln1_rstd, R, d);
    mm_nt_bf16o(A->ln1, W->qkv_w, A->qkv, R, qo, d);           // col-parallel
    k_add_bias_bf(A->qkv, W->qkv_b, R, qo);
    k_split_heads_bf(A->qkv, A->q, A->k, A->v, B, T, Hp, hd);
    flash_attn_fwd_bf(A->q, A->k, A->v, A->atto, F->lse, B, Hp, T, hd, scale);
    k_merge_heads_bf(A->atto, A->atto_m, B, T, Hp, hd);
    mm_nt_bf16o(A->atto_m, W->proj_w, A->proj, R, d, dp);      // row-parallel (partial)
    if (m->allreduce_bf16) m->allreduce_bf16(m->ar_ctx, A->proj, R * d);
    k_bias_residual_bf(A->proj, W->proj_b, x, A->res1, R, d);
    k_layernorm_fwd_bf(A->res1, W->ln2_w, W->ln2_b, A->ln2, F->ln2_mean, F->ln2_rstd, R, d);
    mm_nt_bf16o(A->ln2, W->fc_w, A->fc, R, fo, d);             // col-parallel
    k_bias_gelu_bf(A->fc, W->fc_b, A->fc, A->gelu, R, fo);
    mm_nt_bf16o(A->gelu, W->fcproj_w, A->fcproj, R, d, fo);    // row-parallel (partial)
    if (m->allreduce_bf16) m->allreduce_bf16(m->ar_ctx, A->fcproj, R * d);
    k_bias_residual_bf(A->fcproj, W->fcproj_b, A->res1, A->res2, R, d);
    x = A->res2;
  }
  k_layernorm_fwd_bf(x, w->lnf_w, w->lnf_b, a->lnf, af->lnf_mean, af->lnf_rstd, R, d);
  mm_nt_bf16o(a->lnf, w->wte, a->logits, R, V, d);
  k_cross_entropy_fwd_bf(a->logits, m->d_tgt, af->probs, af->rowloss, R, V);

  float *h = (float *)malloc(R * 4);
  CK(cudaMemcpy(h, af->rowloss, R * 4, cudaMemcpyDeviceToHost));
  double s = 0;
  for (long i = 0; i < R; i++) s += h[i];
  free(h);
  m->loss = (float)(s / R);
  return m->loss;
}

static void dcopyb(void *dst, const void *src, long n) {
  CK(cudaMemcpy(dst, src, n * 2, cudaMemcpyDeviceToDevice));   // bf16 = 2 bytes
}

// BF16 backward: activation grads in bf16 (s_bf), weight grads accumulated fp32 (g)
// for the AdamW master. Mirrors model_backward op-for-op.
void model_backward_bf16(Model *m) {
  Config c = m->cfg;
  int d = c.d_model, H = c.n_head, hd = d / H, V = c.vocab, ff = 4 * d, T = c.seq, B = c.batch;
  long R = (long)B * T, RD = R * d;
  float scale = 1.f / sqrtf((float)hd);
  Weights *w = &m->w_bf, *g = &m->g;
  Acts *a = &m->a_bf, *af = &m->a;
  Bwd *s = &m->s_bf;

  CK(cudaMemset(m->g_arena.base, 0, m->g_arena.off));

  k_cross_entropy_bwd_bf(af->probs, m->d_tgt, s->logits, R, V, 1.f / R);
  mm_nn_bf16o(s->logits, w->wte, s->lnf, R, d, V);
  mm_tn_bf16(s->logits, a->lnf, g->wte, V, d, R);

  void *xlast = c.n_layer ? a->layer[c.n_layer - 1].res2 : a->emb;
  k_layernorm_bwd_bf(s->lnf, xlast, w->lnf_w, af->lnf_mean, af->lnf_rstd,
                     s->dx, g->lnf_w, g->lnf_b, R, d);

  for (int l = c.n_layer - 1; l >= 0; l--) {
    LayerW *W = &w->layer[l], *gL = &g->layer[l];
    LayerAct *A = &a->layer[l], *F = &af->layer[l];
    void *xin = l ? a->layer[l - 1].res2 : a->emb;

    dcopyb(s->res1, s->dx, RD);
    dcopyb(s->fcproj, s->dx, RD);
    k_colsum_bf(s->fcproj, gL->fcproj_b, R, d);
    mm_tn_bf16(s->fcproj, A->gelu, gL->fcproj_w, d, ff, R);
    mm_nn_bf16o(s->fcproj, W->fcproj_w, s->gelu, R, ff, d);
    k_gelu_bwd_bf(A->fc, s->gelu, s->fc, R * ff);
    k_colsum_bf(s->fc, gL->fc_b, R, ff);
    mm_tn_bf16(s->fc, A->ln2, gL->fc_w, ff, d, R);
    mm_nn_bf16o(s->fc, W->fc_w, s->ln2, R, d, ff);
    k_layernorm_bwd_bf(s->ln2, A->res1, W->ln2_w, F->ln2_mean, F->ln2_rstd,
                       s->lntmp, gL->ln2_w, gL->ln2_b, R, d);
    k_add_bf(s->res1, s->lntmp, s->res1, RD);
    dcopyb(s->dx, s->res1, RD);
    dcopyb(s->proj, s->res1, RD);
    k_colsum_bf(s->proj, gL->proj_b, R, d);
    mm_tn_bf16(s->proj, A->atto_m, gL->proj_w, d, d, R);
    mm_nn_bf16o(s->proj, W->proj_w, s->atto_m, R, d, d);
    k_unmerge_heads_bf(s->atto_m, s->atto, B, T, H, hd);
    flash_attn_bwd_bf(A->q, A->k, A->v, A->atto, F->lse, s->atto, s->q, s->k, s->v, B, H, T, hd, scale);
    k_combine_qkv_bf(s->q, s->k, s->v, s->qkv, B, T, H, hd);
    k_colsum_bf(s->qkv, gL->qkv_b, R, 3 * d);
    mm_tn_bf16(s->qkv, A->ln1, gL->qkv_w, 3 * d, d, R);
    mm_nn_bf16o(s->qkv, W->qkv_w, s->ln1, R, d, 3 * d);
    k_layernorm_bwd_bf(s->ln1, xin, W->ln1_w, F->ln1_mean, F->ln1_rstd,
                       s->lntmp, gL->ln1_w, gL->ln1_b, R, d);
    k_add_bf(s->dx, s->lntmp, s->dx, RD);
  }
  k_embed_bwd_wte_bf(s->dx, m->d_idx, g->wte, R, V, d);
  k_embed_bwd_wpe_bf(s->dx, g->wpe, B, T, d);
}

// Tensor-parallel bf16 backward. Conjugate of forward: row-parallel (proj/fcproj)
// bwd is local; column-parallel (qkv/fc) bwd all-reduces the input grad. Replicated
// weights get identical grads on every rank (no comm). tp=1 == model_backward_bf16.
void model_backward_tp(Model *m) {
  Config c = m->cfg;
  int d = c.d_model, H = c.n_head, hd = d / H, V = c.vocab, ff = 4 * d, T = c.seq, B = c.batch;
  int tp = m->tp, dp = d / tp, Hp = H / tp, qo = 3 * d / tp, fo = ff / tp;
  long R = (long)B * T, RD = R * d;
  float scale = 1.f / sqrtf((float)hd);
  Weights *w = &m->w_bf, *g = &m->g;
  Acts *a = &m->a_bf, *af = &m->a;
  Bwd *s = &m->s_bf;

  CK(cudaMemset(m->g_arena.base, 0, m->g_arena.off));

  k_cross_entropy_bwd_bf(af->probs, m->d_tgt, s->logits, R, V, 1.f / R);
  mm_nn_bf16o(s->logits, w->wte, s->lnf, R, d, V);          // d_lnf (replicated)
  mm_tn_bf16(s->logits, a->lnf, g->wte, V, d, R);           // d_wte head part (replicated)

  void *xlast = c.n_layer ? a->layer[c.n_layer - 1].res2 : a->emb;
  k_layernorm_bwd_bf(s->lnf, xlast, w->lnf_w, af->lnf_mean, af->lnf_rstd,
                     s->dx, g->lnf_w, g->lnf_b, R, d);

  for (int l = c.n_layer - 1; l >= 0; l--) {
    LayerW *W = &w->layer[l], *gL = &g->layer[l];
    LayerAct *A = &a->layer[l], *F = &af->layer[l];
    void *xin = l ? a->layer[l - 1].res2 : a->emb;

    dcopyb(s->res1, s->dx, RD);
    dcopyb(s->fcproj, s->dx, RD);
    k_colsum_bf(s->fcproj, gL->fcproj_b, R, d);             // replicated bias grad
    mm_tn_bf16(s->fcproj, A->gelu, gL->fcproj_w, d, fo, R); // sharded weight grad
    mm_nn_bf16o(s->fcproj, W->fcproj_w, s->gelu, R, fo, d); // row-parallel bwd: local
    k_gelu_bwd_bf(A->fc, s->gelu, s->fc, R * fo);
    k_colsum_bf(s->fc, gL->fc_b, R, fo);                    // sharded bias grad
    mm_tn_bf16(s->fc, A->ln2, gL->fc_w, fo, d, R);          // sharded weight grad
    mm_nn_bf16o(s->fc, W->fc_w, s->ln2, R, d, fo);          // col-parallel bwd -> partial d_ln2
    if (m->allreduce_bf16) m->allreduce_bf16(m->ar_ctx, s->ln2, R * d);
    k_layernorm_bwd_bf(s->ln2, A->res1, W->ln2_w, F->ln2_mean, F->ln2_rstd,
                       s->lntmp, gL->ln2_w, gL->ln2_b, R, d);
    k_add_bf(s->res1, s->lntmp, s->res1, RD);
    dcopyb(s->dx, s->res1, RD);
    dcopyb(s->proj, s->res1, RD);
    k_colsum_bf(s->proj, gL->proj_b, R, d);                 // replicated bias grad
    mm_tn_bf16(s->proj, A->atto_m, gL->proj_w, d, dp, R);   // sharded weight grad
    mm_nn_bf16o(s->proj, W->proj_w, s->atto_m, R, dp, d);   // row-parallel bwd: local
    k_unmerge_heads_bf(s->atto_m, s->atto, B, T, Hp, hd);
    flash_attn_bwd_bf(A->q, A->k, A->v, A->atto, F->lse, s->atto, s->q, s->k, s->v, B, Hp, T, hd, scale);
    k_combine_qkv_bf(s->q, s->k, s->v, s->qkv, B, T, Hp, hd);
    k_colsum_bf(s->qkv, gL->qkv_b, R, qo);                  // sharded bias grad
    mm_tn_bf16(s->qkv, A->ln1, gL->qkv_w, qo, d, R);        // sharded weight grad
    mm_nn_bf16o(s->qkv, W->qkv_w, s->ln1, R, d, qo);        // col-parallel bwd -> partial d_ln1
    if (m->allreduce_bf16) m->allreduce_bf16(m->ar_ctx, s->ln1, R * d);
    k_layernorm_bwd_bf(s->ln1, xin, W->ln1_w, F->ln1_mean, F->ln1_rstd,
                       s->lntmp, gL->ln1_w, gL->ln1_b, R, d);
    k_add_bf(s->dx, s->lntmp, s->dx, RD);
  }
  k_embed_bwd_wte_bf(s->dx, m->d_idx, g->wte, R, V, d);
  k_embed_bwd_wpe_bf(s->dx, g->wpe, B, T, d);
}

// ---- pipeline parallel (one stage = this model's layers) ---------------------
// Consume the global weight RNG stream in fixed order; write only the tensors this
// stage owns, so the P stages reassemble to the same model as a single P=1 run.
static void gen_owned(float *dptr_or_null, long n, float std, Rng *st) {
  float *h = (float *)malloc(n * 4);
  for (long i = 0; i < n; i++) h[i] = std * rnd_normal(st);
  if (dptr_or_null) CK(cudaMemcpy(dptr_or_null, h, n * 4, cudaMemcpyHostToDevice));
  free(h);
}
void model_zero_grads(Model *m) { CK(cudaMemset(m->g_arena.base, 0, m->g_arena.off)); }

void model_init_weights_pp(Model *m, uint64_t seed, int lo, int total) {
  Config c = m->cfg;
  int d = c.d_model, V = c.vocab, S = c.seq, ff = 4 * d, nl = c.n_layer;
  Rng st; rng_seed(&st, seed);
  Weights *w = &m->w;
  gen_owned(m->pp_first ? w->wte : NULL, (long)V * d, 0.02f, &st);
  gen_owned(m->pp_first ? w->wpe : NULL, (long)S * d, 0.02f, &st);
  int tp = m->tp, tr = m->rank, dp = d / tp, fo = ff / tp;
  for (int gl = 0; gl < total; gl++) {
    int owned = (gl >= lo && gl < lo + nl);
    LayerW *L = owned ? &w->layer[gl - lo] : NULL;
    if (owned) {                                   // replicated / sharded biases (0)
      fill_const(L->ln1_w, d, 1.f); fill_const(L->ln1_b, d, 0.f);
      fill_const(L->ln2_w, d, 1.f); fill_const(L->ln2_b, d, 0.f);
      fill_const(L->qkv_b, 3 * dp, 0.f); fill_const(L->proj_b, d, 0.f);
      fill_const(L->fc_b, fo, 0.f); fill_const(L->fcproj_b, d, 0.f);
    }
    // double shard: owned layer (PP) + TP slice within the layer
    init_qkv_shard(owned ? L->qkv_w : NULL, d, tp, tr, &st);          // column-parallel
    fill_normal_block(owned ? L->proj_w : NULL, d, d, 0, d, tr * dp, dp, 0.02f, &st);   // row (cols)
    fill_normal_block(owned ? L->fc_w : NULL, ff, d, tr * fo, fo, 0, d, 0.02f, &st);    // column (rows)
    fill_normal_block(owned ? L->fcproj_w : NULL, d, ff, 0, d, tr * fo, fo, 0.02f, &st);// row (cols)
  }
  gen_owned(m->pp_last ? w->head_w : NULL, (long)V * d, 0.02f, &st);
  if (m->pp_last) { fill_const(w->lnf_w, d, 1.f); fill_const(w->lnf_b, d, 0.f); }
}

float model_forward_pp(Model *m, void *xout) {
  Config c = m->cfg;
  int d = c.d_model, H = c.n_head, hd = d / H, V = c.vocab, ff = 4 * d, T = c.seq, B = c.batch;
  int tp = m->tp, dp = d / tp, Hp = H / tp, qo = 3 * d / tp, fo = ff / tp;
  long R = (long)B * T;
  float scale = 1.f / sqrtf((float)hd);
  Weights *w = &m->w_bf;
  Acts *a = &m->a_bf_slot[m->cur_slot], *af = &m->a_slot[m->cur_slot];
  // stage input always lands in this slot's emb buffer (stashed for backward's LN)
  if (m->pp_first) k_embed_bf(w->wte, w->wpe, m->d_idx, a->emb, B, T, d);
  else dcopyb(a->emb, m->pp_xin, R * d);
  void *x = a->emb;
  for (int l = 0; l < c.n_layer; l++) {            // TP-sharded layer (tp=1 -> full width)
    LayerW *W = &w->layer[l];
    LayerAct *A = &a->layer[l], *F = &af->layer[l];
    k_layernorm_fwd_bf(x, W->ln1_w, W->ln1_b, A->ln1, F->ln1_mean, F->ln1_rstd, R, d);
    mm_nt_bf16o(A->ln1, W->qkv_w, A->qkv, R, qo, d);          // column-parallel
    k_add_bias_bf(A->qkv, W->qkv_b, R, qo);
    k_split_heads_bf(A->qkv, A->q, A->k, A->v, B, T, Hp, hd);
    flash_attn_fwd_bf(A->q, A->k, A->v, A->atto, F->lse, B, Hp, T, hd, scale);
    k_merge_heads_bf(A->atto, A->atto_m, B, T, Hp, hd);
    mm_nt_bf16o(A->atto_m, W->proj_w, A->proj, R, d, dp);     // row-parallel (partial)
    if (m->allreduce_bf16) m->allreduce_bf16(m->ar_ctx, A->proj, R * d);
    k_bias_residual_bf(A->proj, W->proj_b, x, A->res1, R, d);
    k_layernorm_fwd_bf(A->res1, W->ln2_w, W->ln2_b, A->ln2, F->ln2_mean, F->ln2_rstd, R, d);
    mm_nt_bf16o(A->ln2, W->fc_w, A->fc, R, fo, d);            // column-parallel
    k_bias_gelu_bf(A->fc, W->fc_b, A->fc, A->gelu, R, fo);
    mm_nt_bf16o(A->gelu, W->fcproj_w, A->fcproj, R, d, fo);   // row-parallel (partial)
    if (m->allreduce_bf16) m->allreduce_bf16(m->ar_ctx, A->fcproj, R * d);
    k_bias_residual_bf(A->fcproj, W->fcproj_b, A->res1, A->res2, R, d);
    x = A->res2;
  }
  if (!m->pp_last) { dcopyb(xout, x, R * d); return 0.f; }
  k_layernorm_fwd_bf(x, w->lnf_w, w->lnf_b, a->lnf, af->lnf_mean, af->lnf_rstd, R, d);
  mm_nt_bf16o(a->lnf, w->head_w, a->logits, R, V, d);
  k_cross_entropy_fwd_bf(a->logits, m->d_tgt, af->probs, af->rowloss, R, V);
  if (m->pp_skip_loss) return 0.f;                 // timing run: avoid the host sync
  float *h = (float *)malloc(R * 4);
  CK(cudaMemcpy(h, af->rowloss, R * 4, cudaMemcpyDeviceToHost));
  double s = 0;
  for (long i = 0; i < R; i++) s += h[i];
  free(h);
  m->loss = (float)(s / R);
  return m->loss;
}

// Backward for one microbatch; grads ACCUMULATE (scheduler zeroes once per step).
// inv_mb = 1/num_microbatches so the summed grad equals the full-batch mean grad.
void model_backward_pp(Model *m, void *dxin, float inv_mb) {
  Config c = m->cfg;
  int d = c.d_model, H = c.n_head, hd = d / H, V = c.vocab, ff = 4 * d, T = c.seq, B = c.batch;
  int tp = m->tp, dp = d / tp, Hp = H / tp, qo = 3 * d / tp, fo = ff / tp;
  long R = (long)B * T, RD = R * d;
  float scale = 1.f / sqrtf((float)hd);
  Weights *w = &m->w_bf, *g = &m->g;
  Acts *a = &m->a_bf_slot[m->cur_slot], *af = &m->a_slot[m->cur_slot];
  Bwd *s = &m->s_bf;

  if (m->pp_last) {
    k_cross_entropy_bwd_bf(af->probs, m->d_tgt, s->logits, R, V, inv_mb / R);
    mm_nn_bf16o(s->logits, w->head_w, s->lnf, R, d, V);
    mm_tn_bf16_acc(s->logits, a->lnf, g->head_w, V, d, R);
    void *xlast = a->layer[c.n_layer - 1].res2;
    k_layernorm_bwd_bf(s->lnf, xlast, w->lnf_w, af->lnf_mean, af->lnf_rstd,
                       s->dx, g->lnf_w, g->lnf_b, R, d);
  } else {
    dcopyb(s->dx, m->pp_dxout, RD);
  }
  for (int l = c.n_layer - 1; l >= 0; l--) {       // TP-sharded backward (tp=1 -> full)
    LayerW *W = &w->layer[l], *gL = &g->layer[l];
    LayerAct *A = &a->layer[l], *F = &af->layer[l];
    void *xin = l ? a->layer[l - 1].res2 : a->emb;   // slot's stashed stage input
    dcopyb(s->res1, s->dx, RD);
    dcopyb(s->fcproj, s->dx, RD);
    k_colsum_bf(s->fcproj, gL->fcproj_b, R, d);
    mm_tn_bf16_acc(s->fcproj, A->gelu, gL->fcproj_w, d, fo, R);
    mm_nn_bf16o(s->fcproj, W->fcproj_w, s->gelu, R, fo, d);   // row-parallel bwd: local
    k_gelu_bwd_bf(A->fc, s->gelu, s->fc, R * fo);
    k_colsum_bf(s->fc, gL->fc_b, R, fo);
    mm_tn_bf16_acc(s->fc, A->ln2, gL->fc_w, fo, d, R);
    mm_nn_bf16o(s->fc, W->fc_w, s->ln2, R, d, fo);            // column-parallel bwd -> partial
    if (m->allreduce_bf16) m->allreduce_bf16(m->ar_ctx, s->ln2, RD);
    k_layernorm_bwd_bf(s->ln2, A->res1, W->ln2_w, F->ln2_mean, F->ln2_rstd,
                       s->lntmp, gL->ln2_w, gL->ln2_b, R, d);
    k_add_bf(s->res1, s->lntmp, s->res1, RD);
    dcopyb(s->dx, s->res1, RD);
    dcopyb(s->proj, s->res1, RD);
    k_colsum_bf(s->proj, gL->proj_b, R, d);
    mm_tn_bf16_acc(s->proj, A->atto_m, gL->proj_w, d, dp, R);
    mm_nn_bf16o(s->proj, W->proj_w, s->atto_m, R, dp, d);     // row-parallel bwd: local
    k_unmerge_heads_bf(s->atto_m, s->atto, B, T, Hp, hd);
    flash_attn_bwd_bf(A->q, A->k, A->v, A->atto, F->lse, s->atto, s->q, s->k, s->v, B, Hp, T, hd, scale);
    k_combine_qkv_bf(s->q, s->k, s->v, s->qkv, B, T, Hp, hd);
    k_colsum_bf(s->qkv, gL->qkv_b, R, qo);
    mm_tn_bf16_acc(s->qkv, A->ln1, gL->qkv_w, qo, d, R);
    mm_nn_bf16o(s->qkv, W->qkv_w, s->ln1, R, d, qo);          // column-parallel bwd -> partial
    if (m->allreduce_bf16) m->allreduce_bf16(m->ar_ctx, s->ln1, RD);
    k_layernorm_bwd_bf(s->ln1, xin, W->ln1_w, F->ln1_mean, F->ln1_rstd,
                       s->lntmp, gL->ln1_w, gL->ln1_b, R, d);
    k_add_bf(s->dx, s->lntmp, s->dx, RD);
  }
  if (m->pp_first) {
    k_embed_bwd_wte_bf(s->dx, m->d_idx, g->wte, R, V, d);
    k_embed_bwd_wpe_bf(s->dx, g->wpe, B, T, d);
  } else {
    dcopyb(dxin, s->dx, RD);
  }
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
