#include "model.h"
#include "infer.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Rollout-throughput benchmark for the M8 engine: tokens/sec at GRPO-relevant group
// sizes, paged-KV memory efficiency vs a naive contiguous cache, and prefix-sharing
// savings when a group shares one prompt. Weights are random (throughput, not quality).

static int has(int c, char **v, const char *k) { for (int i = 1; i < c; i++) if (!strcmp(v[i], k)) return i; return 0; }
static int argi(int c, char **v, const char *k, int d) { int i = has(c, v, k); return i ? atoi(v[i + 1]) : d; }

int main(int argc, char **argv) {
  int L = argi(argc, argv, "--layers", 12);
  int d = argi(argc, argv, "--d", 1024);
  int H = argi(argc, argv, "--heads", 16);
  int V = argi(argc, argv, "--vocab", 50257);
  int seq = argi(argc, argv, "--seq", 1024);
  int plen = argi(argc, argv, "--prompt", 128);
  int n_new = argi(argc, argv, "--new", 256);
  int bs = argi(argc, argv, "--block", 16);
  int bf16 = argi(argc, argv, "--bf16", 1);

  Config cfg = {L, d, H, V, seq, 1};
  Model *m = model_create(cfg);
  model_init_weights(m, 1234);
  if (bf16) model_sync_bf16(m);
  int hd = d / H;
  long es = bf16 ? 2 : 4;

  printf("infer bench: L=%d d=%d H=%d V=%d seq=%d  prompt=%d new=%d block=%d %s\n",
         L, d, H, V, seq, plen, n_new, bs, bf16 ? "bf16" : "fp32");

  int prompt[4096];
  for (int i = 0; i < plen; i++) prompt[i] = (i * 131 + 7) % V;
  int max_lb = (seq + bs - 1) / bs;

  int Gs[] = {1, 8, 16, 32};
  for (int gi = 0; gi < 4; gi++) {
    int G = Gs[gi];
    long need_blocks = (long)G * max_lb + 16;
    Engine *e = infer_create(m, bs, (int)need_blocks, G * max_lb * bs + 64, G + 1, bf16);
    int *out[64], outlen[64];
    for (int i = 0; i < G; i++) out[i] = (int *)malloc((plen + n_new + 1) * 4);
    int *prompts[64], plens[64], news[64];
    for (int i = 0; i < G; i++) { prompts[i] = prompt; plens[i] = plen; news[i] = n_new; }
    InferStats st; float ms = 0;
    long dec = infer_generate(e, G, prompts, plens, news, G, out, outlen, &st, &ms);

    // KV memory: naive reserves seq positions per sequence; paged only the blocks used.
    double per_pos = 2.0 * L * H * hd * es;                  // K+V bytes per token-position
    double naive = (double)G * seq * per_pos;
    double paged = (double)st.blocks_peak * bs * per_pos;
    double tps = dec * 1000.0 / ms;
    printf("  G=%2d: %.0f tok/s (%ld toks, %.1f ms)  KV paged %.1f MB vs naive %.1f MB (%.2fx)  blocks %ld/%ld\n",
           G, tps, dec, ms, paged / 1e6, naive / 1e6, naive / paged, st.blocks_peak, st.blocks_total);
    for (int i = 0; i < G; i++) free(out[i]);
    infer_free(e);
  }

  // Prefix sharing: G samples from one shared prompt (the GRPO rollout pattern).
  for (int gi = 1; gi < 4; gi++) {
    int G = Gs[gi];
    long need_blocks = (long)G * max_lb + 16;
    Engine *e = infer_create(m, bs, (int)need_blocks, plen + 64, G + 1, bf16);
    int *out[64], outlen[64];
    for (int i = 0; i < G; i++) out[i] = (int *)malloc((plen + n_new + 1) * 4);
    InferStats st; float ms = 0;
    infer_generate_group(e, prompt, plen, G, n_new, out, outlen, &st, &ms);
    double per_pos = 2.0 * L * H * hd * es;
    double saved_mb = st.prefix_blocks_saved * bs * per_pos / 1e6;
    printf("  prefix-share G=%2d: blocks saved %ld (%.1f MB), %.0f tok/s  peak %ld/%ld\n",
           G, st.prefix_blocks_saved, saved_mb, st.tokens_decoded * 1000.0 / ms,
           st.blocks_peak, st.blocks_total);
    for (int i = 0; i < G; i++) free(out[i]);
    infer_free(e);
  }
  model_free(m);
  return 0;
}
