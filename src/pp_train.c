#include "model.h"
#include "data.h"
#include "comms.h"
#include "ckpt.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// Pipeline-parallel (P = nranks stages) bf16 training, 1F1B schedule with batched
// (deadlock-free) NCCL send/recv: warmup fill -> steady 1F1B -> cooldown drain.
// Activations stashed per slot (~P microbatches in flight). Measures bubble fraction.
static void tp_ar(void *ctx, void *buf, long n) { comms_tp_allreduce_bf16((Comms *)ctx, buf, n); }
static void dp_rs(void *ctx, const float *s, float *r, long n) { comms_dp_reducescatter_f32((Comms *)ctx, s, r, n); }
static void dp_ag(void *ctx, const float *s, float *r, long n) { comms_dp_allgather_f32((Comms *)ctx, s, r, n); }
static int has(int c, char **v, const char *k) { for (int i = 1; i < c; i++) if (!strcmp(v[i], k)) return i; return 0; }
static int argi(int c, char **v, const char *k, int d) { int i = has(c, v, k); return i ? atoi(v[i + 1]) : d; }
static float argf(int c, char **v, const char *k, float d) { int i = has(c, v, k); return i ? (float)atof(v[i + 1]) : d; }
static const char *args(int c, char **v, const char *k, const char *d) { int i = has(c, v, k); return i ? v[i + 1] : d; }

// One 1F1B step over M microbatches. Grads accumulate; caller zeroes + steps.
static float pp_step(Comms *c, Model *m, int M, int P, int stage, int nslots, long Rd,
                     void *xout, void *dxin, int *mi, int *mt, long R, float inv_mb) {
  int first = m->pp_first, last = m->pp_last;
  int pprev = c->rank - c->tp_size, pnext = c->rank + c->tp_size;   // adjacent-stage peers
  int warmup = P - 1 - stage; if (warmup > M) warmup = M;
  int remaining = M - warmup;
  int ff = 0, bb = 0; float lsum = 0;
  if (!first) comms_recv_bf16(c, m->pp_xin, Rd, pprev);
  for (int i = 0; i < warmup; i++) {                         // warmup: forward only
    model_set_slot(m, ff % nslots); model_set_input(m, mi + (long)ff * R, mt + (long)ff * R);
    float l = model_forward_pp(m, xout); if (last) lsum += l; ff++;
    if (!last) comms_send_bf16(c, xout, Rd, pnext);
    if (!first) comms_recv_bf16(c, m->pp_xin, Rd, pprev);
  }
  for (int i = 0; i < remaining; i++) {                      // steady 1F1B
    model_set_slot(m, ff % nslots); model_set_input(m, mi + (long)ff * R, mt + (long)ff * R);
    float l = model_forward_pp(m, xout); if (last) lsum += l; ff++;
    if (!last) comms_sendrecv_bf16(c, xout, Rd, pnext, m->pp_dxout, Rd, pnext);
    model_set_slot(m, bb % nslots); model_backward_pp(m, dxin, inv_mb); bb++;
    if (i == remaining - 1) { if (!first) comms_send_bf16(c, dxin, Rd, pprev); }
    else if (!first) comms_sendrecv_bf16(c, dxin, Rd, pprev, m->pp_xin, Rd, pprev);
  }
  for (int i = 0; i < warmup; i++) {                         // cooldown: backward only
    if (!last) comms_recv_bf16(c, m->pp_dxout, Rd, pnext);
    model_set_slot(m, bb % nslots); model_backward_pp(m, dxin, inv_mb); bb++;
    if (!first) comms_send_bf16(c, dxin, Rd, pprev);
  }
  comms_sync_default(c);
  return lsum;
}

int main(int argc, char **argv) {
  Comms c;
  comms_init(&c);
  int tp = argi(argc, argv, "--tp", 1);
  int dp = argi(argc, argv, "--dp", 1);
  comms_init_grid3(&c, tp, dp);                      // 3D (TP x PP x DP) grid
  int P = c.pp_size, stage = c.pp_stage;
  int L = argi(argc, argv, "--layers", 8);
  int d = argi(argc, argv, "--d", 512);
  int H = argi(argc, argv, "--heads", 8);
  int seq = argi(argc, argv, "--seq", 256);
  int Bm = argi(argc, argv, "--mbatch", 16);
  int M = argi(argc, argv, "--M", 16);
  int steps = argi(argc, argv, "--steps", 200);
  int logevery = argi(argc, argv, "--log", 50);
  float lr = argf(argc, argv, "--lr", 3e-4f);
  const char *data = args(argc, argv, "--data", "data/input.bin");
  const char *ckpt = args(argc, argv, "--ckpt", NULL);   // per-rank checkpoint prefix
  if (L % P) { if (c.rank == 0) printf("L %% P != 0\n"); comms_finalize(&c); return 1; }

  int nl = L / P, first = (stage == 0), last = (stage == P - 1);
  int num_warmup = P - 1 - stage, nslots = num_warmup + 1;
  Config sc = {nl, d, H, 256, seq, Bm};
  Model *m = model_create_pp(sc, first, last, nslots, tp, c.tp_rank);
  if (tp > 1) { m->allreduce_bf16 = tp_ar; m->ar_ctx = &c; }
  model_init_weights_pp(m, 1337, stage * nl, L);
  if (dp > 1) {                                     // ZeRO-1 optimizer sharding across DP
    model_enable_zero(m, dp, c.dp_rank);
    m->dp_reduce_scatter = dp_rs; m->dp_all_gather = dp_ag; m->dp_ctx = &c;
  }
  if (c.rank == 0)
    printf("TPxPPxDP TP=%d PP=%d DP=%d  L=%d(%d/stage) d=%d h=%d seq=%d mbatch=%d M=%d\n",
           tp, P, dp, L, nl, d, H, seq, Bm, M);

  DataLoader dl;
  data_open(&dl, data, Bm, seq, 42);
  long R = (long)Bm * seq, Rd = R * d;
  int *mi = (int *)malloc((long)M * R * 4), *mt = (int *)malloc((long)M * R * 4);
  for (int j = 0; j < c.dp_rank * M; j++) data_next(&dl, mi, mt);   // skip to this replica's shard
  for (int j = 0; j < M; j++) data_next(&dl, mi + (long)j * R, mt + (long)j * R);
  void *xout = NULL, *dxin = NULL;
  if (!last) CK(cudaMalloc(&xout, Rd * 2));
  if (!first) CK(cudaMalloc(&dxin, Rd * 2));
  float inv_mb = 1.f / M;

  char cpath[512];
  if (ckpt) {                                       // resume bit-exact if this rank's file exists
    snprintf(cpath, sizeof cpath, "%s.r%d", ckpt, c.rank);
    FILE *cf = fopen(cpath, "rb");
    if (cf) { fclose(cf); uint64_t rng; ckpt_load(m, &rng, cpath);
              if (c.rank == 0) printf("resumed from step %d\n", m->step); }
  }

  for (; m->step < steps;) {
    model_sync_bf16(m);
    model_zero_grads(m);
    float lsum = pp_step(&c, m, M, P, stage, nslots, Rd, xout, dxin, mi, mt, R, inv_mb);
    if (dp > 1) model_adamw_step_zero(m, lr, 0.9f, 0.95f, 1e-8f, 0.1f);
    else model_adamw_step(m, lr, 0.9f, 0.95f, 1e-8f, 0.1f);
    if (c.rank == c.nranks - 1 && m->step % logevery == 0) printf("step %5d  loss %.4f\n", m->step, lsum / M);
  }
  if (ckpt) { ckpt_save(m, 0, cpath); comms_sync_default(&c);
              if (c.rank == 0) printf("saved checkpoint at step %d\n", m->step); }

  // bubble: full pipeline step vs M x single-microbatch compute (no comms/pipeline)
  m->pp_skip_loss = 1;
  model_set_input(m, mi, mt);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  for (int i = 0; i < 5; i++) { model_forward_pp(m, xout); model_backward_pp(m, dxin, inv_mb); }
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(a, 0));
  for (int i = 0; i < 20; i++) { model_forward_pp(m, xout); model_backward_pp(m, dxin, inv_mb); }
  CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
  float tmicro = 0; cudaEventElapsedTime(&tmicro, a, b); tmicro /= 20;
  // compute-only microbatch (TP all-reduce disabled) -> isolates TP comms
  void (*ar)(void *, void *, long) = m->allreduce_bf16;
  m->allreduce_bf16 = NULL;
  CK(cudaEventRecord(a, 0));
  for (int i = 0; i < 20; i++) { model_forward_pp(m, xout); model_backward_pp(m, dxin, inv_mb); }
  CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
  float tcomp = 0; cudaEventElapsedTime(&tcomp, a, b); tcomp /= 20;
  m->allreduce_bf16 = ar;
  for (int w = 0; w < 3; w++) pp_step(&c, m, M, P, stage, nslots, Rd, xout, dxin, mi, mt, R, inv_mb);
  CK(cudaEventRecord(a, 0));
  int K = 20;
  for (int k = 0; k < K; k++) pp_step(&c, m, M, P, stage, nslots, Rd, xout, dxin, mi, mt, R, inv_mb);
  CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
  float tstep = 0; cudaEventElapsedTime(&tstep, a, b); tstep /= K;
  // DP optimizer comm: reduce-scatter grads + all-gather params, once per step (ZeRO-1).
  float tdp = 0;
  if (dp > 1) {
    long off = m->zero_off, shard = m->zero_n;
    for (int w = 0; w < 3; w++) {
      m->dp_reduce_scatter(m->dp_ctx, (const float *)m->g_arena.base, m->grad_shard, shard);
      m->dp_all_gather(m->dp_ctx, (const float *)m->w_arena.base + off, (float *)m->w_arena.base, shard);
    }
    CK(cudaDeviceSynchronize()); CK(cudaEventRecord(a, 0));
    for (int k = 0; k < K; k++) {
      m->dp_reduce_scatter(m->dp_ctx, (const float *)m->g_arena.base, m->grad_shard, shard);
      m->dp_all_gather(m->dp_ctx, (const float *)m->w_arena.base + off, (float *)m->w_arena.base, shard);
    }
    CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b)); cudaEventElapsedTime(&tdp, a, b); tdp /= K;
  }
  size_t mfree = 0, mtot = 0; CK(cudaMemGetInfo(&mfree, &mtot));
  if (c.rank == 0) {
    double ideal = (double)M * tmicro, bubble = 1.0 - ideal / tstep;
    double comp = (double)M * tcomp, tpc = (double)M * (tmicro - tcomp), ppc = tstep - ideal;
    double full = tstep + tdp;                                   // full optimizer step
    double toks_local = (double)M * Bm * seq * 1000.0 / full;    // this replica
    double toks_global = toks_local * dp;                        // whole mesh (replicas parallel)
    long Np = 12L * L * d * d + 2L * 256 * d + (long)seq * d;     // ~total params (12d^2/layer)
    double fl = 6.0 * Np * ((double)M * Bm * seq) / (tp * P);     // per-GPU FLOPs/step
    double mfu = fl / (full / 1000.0) / 181.05e12;                // L40S bf16 dense peak
    printf("TP=%d PP=%d DP=%d M=%d: step %.2f ms  bubble %.1f%% (theory=%.1f%%)\n",
           tp, P, dp, M, full, 100.0 * bubble, 100.0 * (P - 1) / (double)(M + P - 1));
    printf("  throughput: %.0f tok/s/replica, %.0f tok/s mesh (%d GPU)  | params ~%.1fM  MFU %.1f%%\n",
           toks_local, toks_global, c.nranks, Np / 1e6, 100.0 * mfu);
    printf("  breakdown/step: compute %.2f | TP all-reduce %.2f | PP bubble+comms %.2f | DP opt-comm %.2f ms\n",
           comp, tpc, ppc, tdp);
    printf("  mem/rank: %.2f / %.2f GB used  | opt-state shard %.0f MB (ZeRO-1 1/%d)\n",
           (mtot - mfree) / 1e9, mtot / 1e9, (m->om_arena.cap + m->ov_arena.cap) / 1e6, dp);
  }

  free(mi); free(mt); data_close(&dl); model_free(m);
  comms_finalize(&c);
  return 0;
}
