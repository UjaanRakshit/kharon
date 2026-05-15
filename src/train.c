#include "model.h"
#include "data.h"
#include "ckpt.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

// BF16 mixed-precision training entrypoint. Trains a proxy GPT on a byte-level
// corpus; logs loss + tokens/sec; checkpoints periodically and resumes bit-exact.
static int has_arg(int c, char **v, const char *k) {
  for (int i = 1; i < c; i++) if (!strcmp(v[i], k)) return i;
  return 0;
}
static int argi(int c, char **v, const char *k, int def) {
  int i = has_arg(c, v, k); return i ? atoi(v[i + 1]) : def;
}
static float argf(int c, char **v, const char *k, float def) {
  int i = has_arg(c, v, k); return i ? (float)atof(v[i + 1]) : def;
}
static const char *args(int c, char **v, const char *k, const char *def) {
  int i = has_arg(c, v, k); return i ? v[i + 1] : def;
}

int main(int argc, char **argv) {
  Config cfg;
  cfg.n_layer = argi(argc, argv, "--layers", 4);
  cfg.d_model = argi(argc, argv, "--d", 256);
  cfg.n_head  = argi(argc, argv, "--heads", 4);
  cfg.vocab   = 256;                       // byte-level
  cfg.seq     = argi(argc, argv, "--seq", 128);
  cfg.batch   = argi(argc, argv, "--batch", 32);
  int steps   = argi(argc, argv, "--steps", 2000);
  int logevery = argi(argc, argv, "--log", 50);
  int ckptevery = argi(argc, argv, "--ckptevery", 500);
  float lr = argf(argc, argv, "--lr", 3e-4f);
  const char *data = args(argc, argv, "--data", "data/input.bin");
  const char *ckpt = args(argc, argv, "--ckpt", "checkpoints/train.ckpt");
  int resume = has_arg(argc, argv, "--resume") != 0;

  Model *m = model_create(cfg);
  DataLoader dl;
  data_open(&dl, data, cfg.batch, cfg.seq, 42);

  uint64_t rng;
  FILE *cf = resume ? fopen(ckpt, "rb") : NULL;
  if (cf) {
    fclose(cf);
    ckpt_load(m, &rng, ckpt);
    dl.rng.s = rng;
    printf("resumed from %s at step %d\n", ckpt, m->step);
  } else {
    model_init_weights(m, 1337);
    printf("fresh init: L=%d d=%d h=%d seq=%d batch=%d vocab=%d (%d params)\n",
           cfg.n_layer, cfg.d_model, cfg.n_head, cfg.seq, cfg.batch, cfg.vocab, m->w.n_param);
  }

  long R = (long)cfg.batch * cfg.seq;
  int *idx = (int *)malloc(R * 4), *tgt = (int *)malloc(R * 4);

  cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
  CK(cudaEventRecord(a, 0));
  int win = 0;
  for (; m->step < steps;) {
    data_next(&dl, idx, tgt);
    model_set_input(m, idx, tgt);
    model_sync_bf16(m);
    float loss = model_forward_bf16(m);
    model_backward_bf16(m);
    model_adamw_step(m, lr, 0.9f, 0.95f, 1e-8f, 0.1f);
    win++;
    if (m->step % logevery == 0) {
      CK(cudaEventRecord(b, 0)); CK(cudaEventSynchronize(b));
      float ms = 0; cudaEventElapsedTime(&ms, a, b);
      double tps = win * R * 1000.0 / ms;
      printf("step %5d  loss %.4f  %.0f tok/s\n", m->step, loss, tps);
      CK(cudaEventRecord(a, 0)); win = 0;
    }
    if (ckptevery && m->step % ckptevery == 0) ckpt_save(m, dl.rng.s, ckpt);
  }
  ckpt_save(m, dl.rng.s, ckpt);
  printf("done at step %d; checkpoint -> %s\n", m->step, ckpt);

  free(idx); free(tgt); data_close(&dl); model_free(m);
  return 0;
}
