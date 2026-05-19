#include "model.h"
#include "data.h"
#include "comms.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// Tensor-parallel (TP=nranks) bf16 training over PCIe. Each rank holds its shard;
// the all-reduce is supplied to the model as a NCCL callback. Logs loss + tokens/sec
// and measures the fraction of step time spent in collectives (the PCIe TP cost).
static void ar_cb(void *ctx, void *buf, long n) { comms_allreduce_bf16((Comms *)ctx, buf, n); }

static int has(int c, char **v, const char *k) { for (int i = 1; i < c; i++) if (!strcmp(v[i], k)) return i; return 0; }
static int argi(int c, char **v, const char *k, int d) { int i = has(c, v, k); return i ? atoi(v[i + 1]) : d; }
static float argf(int c, char **v, const char *k, float d) { int i = has(c, v, k); return i ? (float)atof(v[i + 1]) : d; }
static const char *args(int c, char **v, const char *k, const char *d) { int i = has(c, v, k); return i ? v[i + 1] : d; }

int main(int argc, char **argv) {
  Comms c;
  comms_init(&c);

  Config cfg;
  cfg.n_layer = argi(argc, argv, "--layers", 8);
  cfg.d_model = argi(argc, argv, "--d", 512);
  cfg.n_head  = argi(argc, argv, "--heads", 8);
  cfg.vocab   = 256;
  cfg.seq     = argi(argc, argv, "--seq", 256);
  cfg.batch   = argi(argc, argv, "--batch", 32);
  int steps   = argi(argc, argv, "--steps", 500);
  int logevery = argi(argc, argv, "--log", 50);
  float lr = argf(argc, argv, "--lr", 3e-4f);
  const char *data = args(argc, argv, "--data", "data/input.bin");

  Model *m = model_create_tp(cfg, c.nranks, c.rank);
  m->allreduce_bf16 = ar_cb;
  m->ar_ctx = &c;
  model_init_weights_tp(m, 1337);
  if (c.rank == 0)
    printf("TP=%d  L=%d d=%d h=%d seq=%d batch=%d  (%d params/rank)\n",
           c.nranks, cfg.n_layer, cfg.d_model, cfg.n_head, cfg.seq, cfg.batch, m->w.n_param);

  DataLoader dl;
  data_open(&dl, data, cfg.batch, cfg.seq, 42);   // same seed on all ranks -> same batches
  long R = (long)cfg.batch * cfg.seq;
  int *idx = (int *)malloc(R * 4), *tgt = (int *)malloc(R * 4);

  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  int win = 0; float wt = 0;
  CK(cudaEventRecord(a, 0));
  for (; m->step < steps;) {
    data_next(&dl, idx, tgt);
    model_set_input(m, idx, tgt);
    model_sync_bf16(m);
    float loss = model_forward_tp(m);
    model_backward_tp(m);
    model_adamw_step(m, lr, 0.9f, 0.95f, 1e-8f, 0.1f);
    win++;
    if (m->step % logevery == 0) {
      CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
      float ms = 0; cudaEventElapsedTime(&ms, a, b);
      if (c.rank == 0) printf("step %5d  loss %.4f  %.0f tok/s\n", m->step, loss, win * R * 1000.0 / ms);
      CK(cudaEventRecord(a, 0)); win = 0;
    }
  }

  // comms cost: time full steps, then compute-only steps (all-reduce disabled, timing only)
  int K = 50;
  for (int i = 0; i < 5; i++) { model_sync_bf16(m); model_forward_tp(m); model_backward_tp(m); }
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(a, 0));
  for (int i = 0; i < K; i++) { model_sync_bf16(m); model_forward_tp(m); model_backward_tp(m); }
  CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
  float t_full = 0; cudaEventElapsedTime(&t_full, a, b);
  m->allreduce_bf16 = NULL;                         // disable collectives (timing only)
  CK(cudaEventRecord(a, 0));
  for (int i = 0; i < K; i++) { model_sync_bf16(m); model_forward_tp(m); model_backward_tp(m); }
  CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
  float t_comp = 0; cudaEventElapsedTime(&t_comp, a, b);
  m->allreduce_bf16 = ar_cb;
  (void)wt;
  if (c.rank == 0) {
    double per = t_full / K;
    printf("TP=%d step %.3f ms  compute %.3f ms  comms %.3f ms (%.1f%%)  %.0f tok/s\n",
           c.nranks, per, t_comp / K, (t_full - t_comp) / K,
           100.0 * (t_full - t_comp) / t_full, R * 1000.0 / per);
  }

  free(idx); free(tgt); data_close(&dl); model_free(m);
  comms_finalize(&c);
  return 0;
}
