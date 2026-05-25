#include "model.h"
#include "infer.h"
#include "refio.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// M8 oracle. (1) greedy decode via paged-KV continuous batching == PyTorch greedy
// (gen_ref.bin), token-for-token, across staggered batch entry/exit. (2) paged == a
// contiguous cache (block_size >= max_len) -> same tokens. (3) prefix-sharing group
// generate reproduces the same tokens and reports blocks saved.

static int cmp_tokens(const char *tag, int s, const int *got, int glen, const int *ref, int rlen) {
  int ok = (glen == rlen);
  for (int i = 0; i < glen && i < rlen; i++) if (got[i] != ref[i]) ok = 0;
  printf("  %-22s seq%d len got=%d ref=%d -> %s\n", tag, s, glen, rlen, ok ? "ok" : "FAIL");
  if (!ok) {
    printf("    got:"); for (int i = 0; i < glen; i++) printf(" %d", got[i]); printf("\n");
    printf("    ref:"); for (int i = 0; i < rlen; i++) printf(" %d", ref[i]); printf("\n");
  }
  return ok;
}

// Run the 4 reference prompts through continuous batching at a given block size and
// max_active, comparing each continuation to the reference.
static int run_oracle(Model *m, RefFile *r, int nseq, int block_size, int max_active,
                      int use_bf16, const char *tag) {
  int prompts_buf[8][64]; int *prompts[8], plen[8], n_new[8];
  int *out[8], outlen[8], outbuf[8][128];
  for (int i = 0; i < nseq; i++) {
    char nm[16]; snprintf(nm, sizeof nm, "prompt%d", i);
    RefTensor *pp = ref_get(r, nm);
    snprintf(nm, sizeof nm, "gen%d", i);
    RefTensor *gg = ref_get(r, nm);
    plen[i] = (int)pp->count;
    memcpy(prompts_buf[i], pp->data, plen[i] * 4);
    prompts[i] = prompts_buf[i];
    n_new[i] = (int)gg->count - plen[i];
    out[i] = outbuf[i];
  }
  Engine *e = infer_create(m, block_size, 256, 256, 16, use_bf16);
  InferStats st; float ms;
  infer_generate(e, nseq, prompts, plen, n_new, max_active, out, outlen, &st, &ms);
  int ok = 1;
  for (int i = 0; i < nseq; i++) {
    char nm[16]; snprintf(nm, sizeof nm, "gen%d", i);
    RefTensor *gg = ref_get(r, nm);
    ok &= cmp_tokens(tag, i, out[i], outlen[i], (const int *)gg->data, (int)gg->count);
  }
  printf("  [%s] bs=%d max_active=%d  decoded=%ld blocks_peak=%ld/%ld  %.2f ms\n",
         tag, block_size, max_active, st.tokens_decoded, st.blocks_peak, st.blocks_total, ms);
  infer_free(e);
  return ok;
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "tests/gen_ref.bin";
  RefFile *r = ref_load(path);
  Config cfg = {r->n_layer, r->d_model, r->n_head, r->vocab, r->seq, r->batch};
  Model *m = model_create(cfg);
  model_load_ref(m, r);
  model_sync_bf16(m);
  int nseq = *ref_i32(r, "n_seq");
  int ok = 1;

  printf("fp32 continuous batching vs PyTorch greedy:\n");
  ok &= run_oracle(m, r, nseq, 4, 2, 0, "paged bs=4");          // small blocks, staggered
  printf("paged == contiguous (block_size >= max_len):\n");
  ok &= run_oracle(m, r, nseq, 64, 4, 0, "contiguous");         // one block/seq
  printf("paged == contiguous, all-in-flight:\n");
  ok &= run_oracle(m, r, nseq, 8, 8, 0, "paged bs=8 active=8");

  printf("bf16 throughput path (may differ from fp32 greedy under bf16 rounding):\n");
  run_oracle(m, r, nseq, 8, 4, 1, "bf16 bs=8");                 // reported, not gated

  printf("prefix-sharing group generate (shared prompt, G samples):\n");
  {
    RefTensor *p0 = ref_get(r, "prompt0"); RefTensor *g0 = ref_get(r, "gen0");
    int plen = (int)p0->count, n_new = (int)g0->count - plen, G = 8;
    int prompt[64]; memcpy(prompt, p0->data, plen * 4);
    int *out[8], outlen[8], outbuf[8][128];
    for (int i = 0; i < G; i++) out[i] = outbuf[i];
    Engine *e = infer_create(m, 4, 256, 256, 16, 0);
    InferStats st; float ms;
    infer_generate_group(e, prompt, plen, G, n_new, out, outlen, &st, &ms);
    int gok = 1;
    for (int i = 0; i < G; i++)
      gok &= (outlen[i] == (int)g0->count) &&
             !memcmp(out[i], g0->data, outlen[i] * 4);
    printf("  G=%d all match greedy ref: %s | prefix blocks saved %ld (peak %ld/%ld)  %.2f ms\n",
           G, gok ? "ok" : "FAIL", st.prefix_blocks_saved, st.blocks_peak, st.blocks_total, ms);
    ok &= gok;
    infer_free(e);
  }

  model_free(m); ref_free(r);
  printf("%s\n", ok ? "PASS" : "FAIL");
  return ok ? 0 : 1;
}
