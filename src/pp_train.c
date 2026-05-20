#include "model.h"
#include "data.h"
#include "comms.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// Pipeline-parallel (P = nranks stages) bf16 training with a 1F1B schedule:
// warmup (fill) -> steady 1F1B -> cooldown (drain). Activations stashed per slot
// (~P microbatches in flight), inter-stage transfer via NCCL send/recv. Measures
// the bubble fraction and checks it tracks (P-1)/(M+P-1).
static int has(int c, char **v, const char *k) { for (int i = 1; i < c; i++) if (!strcmp(v[i], k)) return i; return 0; }
static int argi(int c, char **v, const char *k, int d) { int i = has(c, v, k); return i ? atoi(v[i + 1]) : d; }
static float argf(int c, char **v, const char *k, float d) { int i = has(c, v, k); return i ? (float)atof(v[i + 1]) : d; }
static const char *args(int c, char **v, const char *k, const char *d) { int i = has(c, v, k); return i ? v[i + 1] : d; }

int main(int argc, char **argv) {
  Comms c;
  comms_init(&c);
  int P = c.nranks, stage = c.rank;

  int L = argi(argc, argv, "--layers", 8);
  int d = argi(argc, argv, "--d", 512);
  int H = argi(argc, argv, "--heads", 8);
  int seq = argi(argc, argv, "--seq", 256);
  int Bm = argi(argc, argv, "--mbatch", 16);     // microbatch size
  int M = argi(argc, argv, "--M", 16);           // microbatches per step
  int steps = argi(argc, argv, "--steps", 200);
  int logevery = argi(argc, argv, "--log", 50);
  float lr = argf(argc, argv, "--lr", 3e-4f);
  const char *data = args(argc, argv, "--data", "data/input.bin");
  if (L % P) { if (stage == 0) printf("L %% P != 0\n"); comms_finalize(&c); return 1; }

  int nl = L / P, first = (stage == 0), last = (stage == P - 1);
  int num_warmup = P - 1 - stage, nslots = num_warmup + 1;
  Config sc = {nl, d, H, 256, seq, Bm};
  Model *m = model_create_pp(sc, first, last, nslots);
  model_init_weights_pp(m, 1337, stage * nl, L);
  if (stage == 0)
    printf("PP P=%d  L=%d(%d/stage) d=%d h=%d seq=%d mbatch=%d M=%d nslots(s0)=%d\n",
           P, L, nl, d, H, seq, Bm, M, nslots);

  DataLoader dl;
  data_open(&dl, data, Bm, seq, 42);
  long R = (long)Bm * seq, Rd = R * d;
  int *mi = (int *)malloc((long)M * R * 4), *mt = (int *)malloc((long)M * R * 4);
  for (int j = 0; j < M; j++) data_next(&dl, mi + (long)j * R, mt + (long)j * R);
  void *xout = NULL, *dxin = NULL;
  if (!last) CK(cudaMalloc(&xout, Rd * 2));
  if (!first) CK(cudaMalloc(&dxin, Rd * 2));
  float inv_mb = 1.f / M;

#define FWD(j) do { \
    if (!first) comms_recv_bf16(&c, m->pp_xin, Rd, stage - 1); \
    model_set_slot(m, (j) % nslots); model_set_input(m, mi + (long)(j) * R, mt + (long)(j) * R); \
    float _l = model_forward_pp(m, xout); if (last) lsum += _l; \
    if (!last) comms_send_bf16(&c, xout, Rd, stage + 1); \
  } while (0)
#define BWD(j) do { \
    model_set_slot(m, (j) % nslots); \
    if (!last) comms_recv_bf16(&c, m->pp_dxout, Rd, stage + 1); \
    model_backward_pp(m, dxin, inv_mb); \
    if (!first) comms_send_bf16(&c, dxin, Rd, stage - 1); \
  } while (0)

  for (; m->step < steps;) {
    model_sync_bf16(m);
    model_zero_grads(m);
    float lsum = 0; int fwd = 0, bwd = 0;
    for (int i = 0; i < num_warmup; i++) { FWD(fwd); fwd++; }
    while (fwd < M) { FWD(fwd); fwd++; BWD(bwd); bwd++; }
    while (bwd < M) { BWD(bwd); bwd++; }
    comms_sync_default(&c);
    model_adamw_step(m, lr, 0.9f, 0.95f, 1e-8f, 0.1f);
    if (last && m->step % logevery == 0) printf("step %5d  loss %.4f\n", m->step, lsum / M);
  }

  // bubble measurement: time the full pipeline step vs M x single-microbatch compute
  m->pp_skip_loss = 1;
  model_set_input(m, mi, mt);
  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  // per-microbatch compute (one stage, no pipeline/comms)
  for (int i = 0; i < 5; i++) { model_forward_pp(m, xout); model_backward_pp(m, dxin, inv_mb); }
  CK(cudaDeviceSynchronize());
  CK(cudaEventRecord(a, 0));
  for (int i = 0; i < 20; i++) { model_forward_pp(m, xout); model_backward_pp(m, dxin, inv_mb); }
  CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
  float tmicro = 0; cudaEventElapsedTime(&tmicro, a, b); tmicro /= 20;
  // full pipeline step
  int K = 20; float lsum = 0; (void)lsum;
  for (int w = 0; w < 3; w++) { int fwd = 0, bwd = 0;
    for (int i = 0; i < num_warmup; i++) { FWD(fwd); fwd++; }
    while (fwd < M) { FWD(fwd); fwd++; BWD(bwd); bwd++; } while (bwd < M) { BWD(bwd); bwd++; }
    comms_sync_default(&c); }
  CK(cudaEventRecord(a, 0));
  for (int k = 0; k < K; k++) { int fwd = 0, bwd = 0;
    for (int i = 0; i < num_warmup; i++) { FWD(fwd); fwd++; }
    while (fwd < M) { FWD(fwd); fwd++; BWD(bwd); bwd++; } while (bwd < M) { BWD(bwd); bwd++; }
    comms_sync_default(&c); }
  CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
  float tstep = 0; cudaEventElapsedTime(&tstep, a, b); tstep /= K;
  if (stage == 0) {
    double ideal = (double)M * tmicro, bubble = 1.0 - ideal / tstep;
    printf("P=%d M=%d: step %.2f ms  ideal %.2f ms  bubble %.1f%% (theory (P-1)/(M+P-1)=%.1f%%)\n",
           P, M, tstep, ideal, 100.0 * bubble, 100.0 * (P - 1) / (double)(M + P - 1));
  }

  free(mi); free(mt); data_close(&dl); model_free(m);
  comms_finalize(&c);
  return 0;
}
