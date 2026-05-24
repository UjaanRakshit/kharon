#include "infer.h"
#include "kernels.h"
#include "paged.h"
#include "kharon.h"
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// Paged-KV inference engine. The decode forward mirrors model_forward_bf16 but reads
// q/k/v straight from the fused qkv buffer (no head transpose), writes k/v into a paged
// cache, and runs causal paged attention. fp32 path = token-exact oracle; bf16 = speed.

struct Engine {
  Model *m;
  int use_bf16, es;                 // element size 4/2
  int d, H, hd, V, n_layer, ff;
  int bs, n_blocks, max_lb, max_tokens, max_seqs;
  void *kc, *vc;                    // [n_layer, n_blocks, bs, H, hd]
  long lstride;                     // elements per layer in cache
  // device step buffers (es-sized)
  void *x, *ln1, *qkv, *atto, *proj, *res1, *ln2, *fc, *gelu, *fcproj, *res2, *lnf;
  void *gath, *logits;
  float *mean, *rstd;
  int *d_tok, *d_pos, *d_seqslot, *d_btab, *d_rows;
  float *h_logits;                  // host, fp32
  // block allocator
  int *freelist, n_free;
  long blocks_peak, blocks_inuse;
};

static void *dalloc(long bytes) { void *p; CK(cudaMalloc(&p, bytes)); return p; }

Engine *infer_create(Model *m, int block_size, int n_blocks, int max_tokens,
                     int max_seqs, int use_bf16) {
  Engine *e = (Engine *)calloc(1, sizeof(Engine));
  e->m = m; e->use_bf16 = use_bf16; e->es = use_bf16 ? 2 : 4;
  Config c = m->cfg;
  e->d = c.d_model; e->H = c.n_head; e->hd = c.d_model / c.n_head; e->V = c.vocab;
  e->n_layer = c.n_layer; e->ff = 4 * c.d_model;
  e->bs = block_size; e->n_blocks = n_blocks;
  e->max_lb = (c.seq + block_size - 1) / block_size;
  e->max_tokens = max_tokens; e->max_seqs = max_seqs;
  if (e->hd > 1024) DIE("infer: hd %d too large for one block", e->hd);

  e->lstride = (long)n_blocks * block_size * e->H * e->hd;
  e->kc = dalloc((long)e->n_layer * e->lstride * e->es);
  e->vc = dalloc((long)e->n_layer * e->lstride * e->es);

  long mt = max_tokens, d = e->d, ff = e->ff, es = e->es;
  e->x = dalloc(mt * d * es);   e->ln1 = dalloc(mt * d * es);   e->qkv = dalloc(mt * 3 * d * es);
  e->atto = dalloc(mt * d * es); e->proj = dalloc(mt * d * es); e->res1 = dalloc(mt * d * es);
  e->ln2 = dalloc(mt * d * es); e->fc = dalloc(mt * ff * es);   e->gelu = dalloc(mt * ff * es);
  e->fcproj = dalloc(mt * d * es); e->res2 = dalloc(mt * d * es); e->lnf = dalloc(mt * d * es);
  e->gath = dalloc((long)max_seqs * d * es); e->logits = dalloc((long)max_seqs * e->V * es);
  e->mean = (float *)dalloc(mt * 4); e->rstd = (float *)dalloc(mt * 4);
  e->d_tok = (int *)dalloc(mt * 4); e->d_pos = (int *)dalloc(mt * 4);
  e->d_seqslot = (int *)dalloc(mt * 4);
  e->d_btab = (int *)dalloc((long)max_seqs * e->max_lb * 4);
  e->d_rows = (int *)dalloc((long)max_seqs * 4);
  e->h_logits = (float *)malloc((long)max_seqs * e->V * 4);

  e->freelist = (int *)malloc(n_blocks * 4);
  for (int i = 0; i < n_blocks; i++) e->freelist[i] = n_blocks - 1 - i;  // pop low ids first
  e->n_free = n_blocks;
  gemm_init();
  return e;
}

void infer_free(Engine *e) {
  if (!e) return;
  void *bufs[] = {e->kc, e->vc, e->x, e->ln1, e->qkv, e->atto, e->proj, e->res1, e->ln2,
                  e->fc, e->gelu, e->fcproj, e->res2, e->lnf, e->gath, e->logits,
                  e->mean, e->rstd, e->d_tok, e->d_pos, e->d_seqslot, e->d_btab, e->d_rows};
  for (size_t i = 0; i < sizeof(bufs) / sizeof(*bufs); i++) cudaFree(bufs[i]);
  free(e->h_logits); free(e->freelist); free(e);
}

static int blk_alloc(Engine *e) {
  if (e->n_free == 0) DIE("infer: KV cache out of blocks (%ld)", (long)e->n_blocks);
  int pb = e->freelist[--e->n_free];
  if (++e->blocks_inuse > e->blocks_peak) e->blocks_peak = e->blocks_inuse;
  return pb;
}
static void blk_free(Engine *e, int pb) { e->freelist[e->n_free++] = pb; e->blocks_inuse--; }

// Device-to-device copy of one physical block (K and V, all layers). Used to give each
// shared-prefix sequence a private copy of the partial last block (copy-on-write).
static void blk_copy(Engine *e, int src, int dst) {
  long blkelems = (long)e->bs * e->H * e->hd, bytes = blkelems * e->es;
  for (int l = 0; l < e->n_layer; l++) {
    long ko = ((long)l * e->lstride + (long)src * blkelems) * e->es;
    long ndst = ((long)l * e->lstride + (long)dst * blkelems) * e->es;
    CK(cudaMemcpy((char *)e->kc + ndst, (char *)e->kc + ko, bytes, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy((char *)e->vc + ndst, (char *)e->vc + ko, bytes, cudaMemcpyDeviceToDevice));
  }
}

// ---- per-op dtype wrappers ---------------------------------------------------
static void op_embed(Engine *e, void *wte, void *wpe, int *tok, int *pos, void *out, long n) {
  if (e->use_bf16) k_embed_pos_bf(wte, wpe, tok, pos, out, e->d, n);
  else k_embed_pos((float *)wte, (float *)wpe, tok, pos, (float *)out, e->d, n);
}
static void op_ln(Engine *e, void *x, void *w, void *b, void *out, int rows) {
  if (e->use_bf16) k_layernorm_fwd_bf(x, w, b, out, e->mean, e->rstd, rows, e->d);
  else k_layernorm_fwd((float *)x, (float *)w, (float *)b, (float *)out, e->mean, e->rstd, rows, e->d);
}
static void op_mm(Engine *e, void *A, void *B, void *C, int M, int N, int K) {
  if (e->use_bf16) mm_nt_bf16o(A, B, C, M, N, K);
  else mm_nt((float *)A, (float *)B, (float *)C, M, N, K);
}
static void op_addbias(Engine *e, void *y, void *b, int rows, int N) {
  if (e->use_bf16) k_add_bias_bf(y, b, rows, N);
  else k_add_bias((float *)y, (float *)b, rows, N);
}
static void op_biasres(Engine *e, void *y, void *bias, void *resid, void *out, int rows, int N) {
  if (e->use_bf16) k_bias_residual_bf(y, bias, resid, out, rows, N);
  else k_bias_residual((float *)y, (float *)bias, (float *)resid, (float *)out, rows, N);
}
static void op_biasgelu(Engine *e, void *y, void *bias, void *pre, void *act, int rows, int N) {
  if (e->use_bf16) k_bias_gelu_bf(y, bias, pre, act, rows, N);
  else k_bias_gelu((float *)y, (float *)bias, (float *)pre, (float *)act, rows, N);
}
static void op_append(Engine *e, void *qkv, void *kc, void *vc, long ntok) {
  if (e->use_bf16) k_append_kv_bf(qkv, kc, vc, e->d_seqslot, e->d_pos, e->d_btab,
                                  e->max_lb, e->bs, e->H, e->hd, e->d, ntok);
  else k_append_kv((float *)qkv, (float *)kc, (float *)vc, e->d_seqslot, e->d_pos, e->d_btab,
                   e->max_lb, e->bs, e->H, e->hd, e->d, ntok);
}
static void op_attn(Engine *e, void *qkv, void *kc, void *vc, float scale, void *out, long ntok) {
  if (e->use_bf16) k_paged_attn_bf(qkv, kc, vc, e->d_seqslot, e->d_pos, e->d_btab, e->max_lb,
                                   e->bs, e->H, e->hd, e->d, scale, out, ntok);
  else k_paged_attn((float *)qkv, (float *)kc, (float *)vc, e->d_seqslot, e->d_pos, e->d_btab,
                    e->max_lb, e->bs, e->H, e->hd, e->d, scale, (float *)out, ntok);
}
static void op_gather(Engine *e, void *src, int *rows, void *dst, int nrow) {
  if (e->use_bf16) k_gather_rows_bf(src, rows, dst, e->d, nrow);
  else k_gather_rows((float *)src, rows, (float *)dst, e->d, nrow);
}

// One forward over ntok query tokens for nseq active sequences. Logits for the last
// token of each sequence land in e->h_logits[seq*V .. ]. Metadata must be on device.
static void run_step(Engine *e, int ntok, int nseq) {
  int d = e->d, ff = e->ff, V = e->V;
  float scale = 1.f / sqrtf((float)e->hd);
  Weights *w = e->use_bf16 ? &e->m->w_bf : &e->m->w;
  op_embed(e, w->wte, w->wpe, e->d_tok, e->d_pos, e->x, ntok);
  void *x = e->x;
  for (int l = 0; l < e->n_layer; l++) {
    LayerW *L = &w->layer[l];
    void *kc = (char *)e->kc + (long)l * e->lstride * e->es;
    void *vc = (char *)e->vc + (long)l * e->lstride * e->es;
    op_ln(e, x, L->ln1_w, L->ln1_b, e->ln1, ntok);
    op_mm(e, e->ln1, L->qkv_w, e->qkv, ntok, 3 * d, d);
    op_addbias(e, e->qkv, L->qkv_b, ntok, 3 * d);
    op_append(e, e->qkv, kc, vc, ntok);
    op_attn(e, e->qkv, kc, vc, scale, e->atto, ntok);
    op_mm(e, e->atto, L->proj_w, e->proj, ntok, d, d);
    op_biasres(e, e->proj, L->proj_b, x, e->res1, ntok, d);
    op_ln(e, e->res1, L->ln2_w, L->ln2_b, e->ln2, ntok);
    op_mm(e, e->ln2, L->fc_w, e->fc, ntok, ff, d);
    op_biasgelu(e, e->fc, L->fc_b, e->fc, e->gelu, ntok, ff);
    op_mm(e, e->gelu, L->fcproj_w, e->fcproj, ntok, d, ff);
    op_biasres(e, e->fcproj, L->fcproj_b, e->res1, e->res2, ntok, d);
    x = e->res2;
  }
  op_ln(e, x, w->lnf_w, w->lnf_b, e->lnf, ntok);
  op_gather(e, e->lnf, e->d_rows, e->gath, nseq);
  op_mm(e, e->gath, w->wte, e->logits, nseq, V, d);    // weight-tied head
  CK(cudaDeviceSynchronize());
}

// Copy the nseq logit rows to the host fp32 staging buffer (widening bf16).
static void pull_logits(Engine *e, int nseq) {
  long n = (long)nseq * e->V;
  if (e->use_bf16) {
    unsigned short *tmp = (unsigned short *)malloc(n * 2);
    CK(cudaMemcpy(tmp, e->logits, n * 2, cudaMemcpyDeviceToHost));
    for (long i = 0; i < n; i++) {
      unsigned int bits = (unsigned int)tmp[i] << 16; float f; memcpy(&f, &bits, 4); e->h_logits[i] = f;
    }
    free(tmp);
  } else {
    CK(cudaMemcpy(e->h_logits, e->logits, n * 4, cudaMemcpyDeviceToHost));
  }
}

static int argmax_row(const float *r, int V) {
  int best = 0; float bv = r[0];
  for (int j = 1; j < V; j++) if (r[j] > bv) { bv = r[j]; best = j; }
  return best;
}

// Per-sequence runtime state for the scheduler.
typedef struct {
  int *tok, len, cached, target, plen, done, active, slot;
  int *btab, nblk;     // logical->physical block table (host)
  int *out; int *outlen;
} Seq;

static void ensure_blocks(Engine *e, Seq *s, int upto_pos) {
  int need = upto_pos / e->bs + 1;
  while (s->nblk < need) s->btab[s->nblk++] = blk_alloc(e);
}

// Drive `na` active sequences through one continuous-batch step: build the flat query
// token list (uncached positions per seq), run the forward, greedy-sample, append.
static void batch_step(Engine *e, Seq **act, int na) {
  int V = e->V, mlb = e->max_lb;
  int *h_tok = (int *)malloc(e->max_tokens * 4), *h_pos = (int *)malloc(e->max_tokens * 4);
  int *h_slot = (int *)malloc(e->max_tokens * 4), *h_rows = (int *)malloc(na * 4);
  int *h_btab = (int *)malloc((long)na * mlb * 4);
  int ntok = 0;
  for (int i = 0; i < na; i++) {
    Seq *s = act[i];
    for (int p = s->cached; p < s->len; p++) {       // uncached known positions
      h_tok[ntok] = s->tok[p]; h_pos[ntok] = p; h_slot[ntok] = i;
      if (p == s->len - 1) h_rows[i] = ntok;
      ntok++;
    }
    for (int b = 0; b < mlb; b++)
      h_btab[(long)i * mlb + b] = (b < s->nblk) ? s->btab[b] : 0;
  }
  CK(cudaMemcpy(e->d_tok, h_tok, ntok * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(e->d_pos, h_pos, ntok * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(e->d_seqslot, h_slot, ntok * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(e->d_btab, h_btab, (long)na * mlb * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(e->d_rows, h_rows, na * 4, cudaMemcpyHostToDevice));

  run_step(e, ntok, na);
  pull_logits(e, na);
  for (int i = 0; i < na; i++) {
    Seq *s = act[i];
    s->cached = s->len;                              // positions [.., len-1] now cached
    int nxt = argmax_row(e->h_logits + (long)i * V, V);
    s->tok[s->len] = nxt; s->len++;
    if (s->len - 1 < mlb * e->bs) ensure_blocks(e, s, s->len - 1);
    if (s->len >= s->target) s->done = 1;
  }
  free(h_tok); free(h_pos); free(h_slot); free(h_rows); free(h_btab);
}

long infer_generate(Engine *e, int nseq, int **prompts, int *plen, int *n_new,
                    int max_active, int **out, int *outlen, InferStats *st, float *ms) {
  Seq *seqs = (Seq *)calloc(nseq, sizeof(Seq));
  for (int i = 0; i < nseq; i++) {
    Seq *s = &seqs[i];
    s->target = plen[i] + n_new[i]; s->plen = plen[i];
    s->tok = (int *)malloc((s->target + 1) * 4);
    memcpy(s->tok, prompts[i], plen[i] * 4);
    s->len = plen[i]; s->cached = 0; s->done = 0;
    s->btab = (int *)malloc(e->max_lb * 4); s->nblk = 0;
    s->out = out[i];
  }
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  CK(cudaEventRecord(a, 0));
  long decoded = 0;
  int admitted = 0, finished = 0;
  Seq *act[256];
  while (finished < nseq) {
    int na = 0;                                      // build active set (continuous batching)
    for (int i = 0; i < nseq && na < max_active; i++)
      if (!seqs[i].done && seqs[i].active) act[na++] = &seqs[i];
    while (na < max_active && admitted < nseq) {     // admit waiting seqs into free slots
      if (!seqs[admitted].active && !seqs[admitted].done) {
        seqs[admitted].active = 1;
        ensure_blocks(e, &seqs[admitted], seqs[admitted].len - 1);  // prompt blocks on entry
        act[na++] = &seqs[admitted];
      }
      admitted++;
    }
    if (na == 0) break;
    batch_step(e, act, na);
    decoded += na;                                   // one token per active seq per step
    for (int i = 0; i < nseq; i++) {
      if (seqs[i].active && seqs[i].done && seqs[i].out) {
        Seq *s = &seqs[i];
        memcpy(s->out, s->tok, s->len * 4); if (outlen) outlen[i] = s->len;
        for (int bb = 0; bb < s->nblk; bb++) blk_free(e, s->btab[bb]);
        s->nblk = 0; s->active = 0; finished++;
        s->out = NULL;                               // mark collected
      }
    }
  }
  CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
  if (ms) { *ms = 0; cudaEventElapsedTime(ms, a, b); }
  if (st) { st->blocks_total = e->n_blocks; st->blocks_peak = e->blocks_peak;
            st->tokens_decoded = decoded; st->prefix_blocks_saved = 0; }
  for (int i = 0; i < nseq; i++) { free(seqs[i].tok); free(seqs[i].btab); }
  free(seqs);
  return decoded;
}

long infer_generate_group(Engine *e, int *prompt, int plen, int G, int n_new,
                          int **out, int *outlen, InferStats *st, float *ms) {
  // Prefill the shared prompt once into prefix blocks, then share them across G seqs.
  int prefix_nblk = (plen - 1) / e->bs + 1;
  int *prefix_btab = (int *)malloc(prefix_nblk * 4);
  for (int b = 0; b < prefix_nblk; b++) prefix_btab[b] = blk_alloc(e);

  // single-sequence prefill of the prompt
  int h_tok[4096], h_pos[4096], h_slot[4096], h_rows[1], *h_btab;
  h_btab = (int *)malloc(e->max_lb * 4);
  for (int p = 0; p < plen; p++) { h_tok[p] = prompt[p]; h_pos[p] = p; h_slot[p] = 0; }
  h_rows[0] = plen - 1;
  for (int b = 0; b < e->max_lb; b++) h_btab[b] = (b < prefix_nblk) ? prefix_btab[b] : 0;
  CK(cudaMemcpy(e->d_tok, h_tok, plen * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(e->d_pos, h_pos, plen * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(e->d_seqslot, h_slot, plen * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(e->d_btab, h_btab, e->max_lb * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(e->d_rows, h_rows, 4, cudaMemcpyHostToDevice));
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  CK(cudaEventRecord(a, 0));
  run_step(e, plen, 1);
  pull_logits(e, 1);                                   // prefill logits -> host for sampling
  // Only FULL prefix blocks can be shared read-only; the partial last block holds prefix
  // tokens that sit in the same block as the first generated token, so each sequence gets
  // a private copy-on-write of it (classic vLLM behaviour).
  int full = plen / e->bs, partial = plen % e->bs;
  Seq *seqs = (Seq *)calloc(G, sizeof(Seq));
  for (int i = 0; i < G; i++) {
    Seq *s = &seqs[i];
    s->target = plen + n_new; s->plen = plen;
    s->tok = (int *)malloc((s->target + 1) * 4);
    memcpy(s->tok, prompt, plen * 4);
    s->len = plen; s->cached = plen; s->done = 0; s->active = 1;
    s->btab = (int *)malloc(e->max_lb * 4);
    for (int bb = 0; bb < full; bb++) s->btab[bb] = prefix_btab[bb];   // shared full blocks
    s->nblk = full;
    if (partial) {                                     // private copy of the partial block
      int pv = blk_alloc(e);
      blk_copy(e, prefix_btab[full], pv);
      s->btab[s->nblk++] = pv;
    }
    s->out = out[i];
    int nxt = argmax_row(e->h_logits, e->V);            // first token from prefill logits
    s->tok[s->len] = nxt; s->len++;
    if (s->len - 1 < e->max_lb * e->bs) ensure_blocks(e, s, s->len - 1);
  }
  long saved = (long)(G - 1) * full;                    // full prefix blocks stored once
  Seq *act[256];
  long decoded = (long)G;                              // the first sampled token each
  int finished = 0;
  while (finished < G) {
    int na = 0;
    for (int i = 0; i < G; i++) if (!seqs[i].done) act[na++] = &seqs[i];
    if (na == 0) break;
    batch_step(e, act, na);
    decoded += na;
    for (int i = 0; i < G; i++)
      if (seqs[i].done && seqs[i].out) {
        Seq *s = &seqs[i];
        memcpy(s->out, s->tok, s->len * 4); if (outlen) outlen[i] = s->len;
        for (int bb = full; bb < s->nblk; bb++) blk_free(e, s->btab[bb]);  // private blocks only
        s->out = NULL; finished++;
      }
  }
  CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
  if (ms) { *ms = 0; cudaEventElapsedTime(ms, a, b); }
  for (int b2 = 0; b2 < prefix_nblk; b2++) blk_free(e, prefix_btab[b2]);
  if (st) { st->blocks_total = e->n_blocks; st->blocks_peak = e->blocks_peak;
            st->tokens_decoded = decoded; st->prefix_blocks_saved = saved; }
  for (int i = 0; i < G; i++) { free(seqs[i].tok); free(seqs[i].btab); }
  free(seqs); free(prefix_btab); free(h_btab);
  return decoded;
}
