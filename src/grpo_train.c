#include "model.h"
#include "infer.h"
#include "grpo.h"
#include "kernels.h"
#include "rng.h"
#include "kharon.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

// GRPO on a verifiable task: two-digit addition, byte-level (vocab 256). Prompt "a+b="
// (4 chars), the policy samples G completions of 2 chars, reward = exact match of the
// zero-padded sum. Advantages are group-relative (no critic); the update reuses the
// trainer's backward via model_grpo_backward with a KL penalty to the frozen init policy.
// Rollouts come from the M8 engine (prefix-shared per group). Reward curve is the deliverable.

static int has(int c, char **v, const char *k) { for (int i = 1; i < c; i++) if (!strcmp(v[i], k)) return i; return 0; }
static int argi(int c, char **v, const char *k, int d) { int i = has(c, v, k); return i ? atoi(v[i + 1]) : d; }
static float argf(int c, char **v, const char *k, float d) { int i = has(c, v, k); return i ? (float)atof(v[i + 1]) : d; }

#define PLEN 4
#define NNEW 2
#define T (PLEN + NNEW)

// "a+b=" -> 4 prompt bytes; correct answer is the zero-padded 2-char sum.
static void make_prompt(int a, int b, int *p, char *ans) {
  p[0] = '0' + a; p[1] = '+'; p[2] = '0' + b; p[3] = '=';
  int s = a + b; ans[0] = '0' + s / 10; ans[1] = '0' + s % 10;
}
static int exact(const int *completion, const char *ans) {
  return completion[0] == ans[0] && completion[1] == ans[1];
}
// Dense, programmatically-verifiable reward (shaping beats sparse exact-match cold-start):
// + format credit for emitting digits, + closeness of the parsed number to the true sum,
// + a bonus for an exact match. All three are checkable, no learned reward model.
static float reward(const int *completion, const char *ans, int sum) {
  int c0 = completion[0], c1 = completion[1];
  int d0 = (c0 >= '0' && c0 <= '9'), d1 = (c1 >= '0' && c1 <= '9');
  float r = 0.1f * d0 + 0.1f * d1;             // format: emit digits
  if (d0 && d1) {
    int pred = (c0 - '0') * 10 + (c1 - '0');
    int e = pred - sum; if (e < 0) e = -e;
    r += 0.3f * (1.0f - (e > 18 ? 18 : e) / 18.0f);  // weak closeness nudge
    if (pred == sum) r += 2.0f;                // exact match dominates (anti reward-hack)
  }
  return r;
}

// Greedy held-out accuracy over all 100 (a,b) problems. show>0 prints sample completions
// (so reward hacking — e.g. always emitting one "safe" number — is visible, not hidden).
static float eval_accuracy(Engine *e, int show) {
  infer_set_sampling(e, 0.f, 1);                 // greedy
  int correct = 0, shown = 0;
  for (int a = 0; a < 10; a++) for (int b = 0; b < 10; b++) {
    int prompt[PLEN]; char ans[2]; make_prompt(a, b, prompt, ans);
    int outbuf[T]; int *out[1] = {outbuf}; int outlen[1];
    int *prompts[1] = {prompt}; int plen[1] = {PLEN}, nnew[1] = {NNEW};
    InferStats st; float ms;
    infer_generate(e, 1, prompts, plen, nnew, 1, out, outlen, &st, &ms);
    if (exact(outbuf + PLEN, ans)) correct++;
    if (show && shown < show && (a + b) % 3 == 0) {
      printf("    %d+%d=%c%c (want %c%c)\n", a, b, outbuf[PLEN], outbuf[PLEN + 1], ans[0], ans[1]);
      shown++;
    }
  }
  return correct / 100.f;
}

int main(int argc, char **argv) {
  int L = argi(argc, argv, "--layers", 4);
  int d = argi(argc, argv, "--d", 256);
  int H = argi(argc, argv, "--heads", 8);
  int G = argi(argc, argv, "--G", 8);
  int NP = argi(argc, argv, "--prompts", 16);
  int steps = argi(argc, argv, "--steps", 400);
  float lr = argf(argc, argv, "--lr", 1e-3f);
  float temp = argf(argc, argv, "--temp", 1.0f);
  float beta = argf(argc, argv, "--beta", 0.02f);
  int logevery = argi(argc, argv, "--log", 20);
  int B = NP * G;

  Config cfg = {L, d, H, 256, T, B};
  Model *m = model_create(cfg);
  model_init_weights(m, 1337);
  Model *ref = model_create(cfg);                // frozen reference policy (same init)
  model_init_weights(ref, 1337);
  model_sync_bf16(ref);

  Engine *e = infer_create(m, 2, B * T + 64, B * T + 64, B + 1, 1);  // bs=2: prompt = 2 shared blocks
  Rng task; rng_seed(&task, 20260530);

  int *idx = (int *)malloc((long)B * T * 4), *tgt = (int *)malloc((long)B * T * 4);
  float *adv = (float *)malloc((long)B * 4), *rew = (float *)malloc((long)B * 4);
  float *h_coef = (float *)malloc((long)B * T * 4);
  float *curp = (float *)malloc((long)B * T * 4), *refp = (float *)malloc((long)B * T * 4);
  float *d_coef, *d_curp, *d_refp;
  CK(cudaMalloc(&d_coef, (long)B * T * 4));
  CK(cudaMalloc(&d_curp, (long)B * T * 4)); CK(cudaMalloc(&d_refp, (long)B * T * 4));
  char (*ans)[2] = (char (*)[2])malloc((long)NP * 2);

  printf("GRPO 2-digit add: L=%d d=%d H=%d  G=%d prompts=%d (B=%d) T=%d  lr=%g temp=%g beta=%g\n",
         L, d, H, G, NP, B, T, lr, temp, beta);
  model_sync_bf16(m);
  printf("step %4d  acc %.3f (init)\n", 0, eval_accuracy(e, 0));

  for (int step = 1; step <= steps; step++) {
    model_sync_bf16(m);                          // engine serves the current policy
    infer_set_sampling(e, temp, 0x1234ull * step + 1);
    double roll_ms = 0, roll_tok = 0, roll_saved = 0;
    // rollout: G samples per prompt (prefix-shared), fill the [B,T] training batch
    for (int pi = 0; pi < NP; pi++) {
      int a = rng_u32(&task) % 10, b = rng_u32(&task) % 10;
      int prompt[PLEN]; make_prompt(a, b, prompt, ans[pi]);
      int *out[64], outlen[64], outbuf[64][T];
      for (int g = 0; g < G; g++) out[g] = outbuf[g];
      InferStats st; float ms;
      infer_generate_group(e, prompt, PLEN, G, NNEW, out, outlen, &st, &ms);
      roll_ms += ms; roll_tok += st.tokens_decoded; roll_saved += st.prefix_blocks_saved;
      for (int g = 0; g < G; g++) {
        int row = pi * G + g;
        rew[row] = reward(outbuf[g] + PLEN, ans[pi], a + b);
        for (int t = 0; t < T; t++) {
          idx[(long)row * T + t] = outbuf[g][t];
          tgt[(long)row * T + t] = (t < T - 1) ? outbuf[g][t + 1] : 0;
        }
      }
    }
    grpo_advantages(rew, NP, G, 1e-4f, adv);

    // current-policy + reference forward, gather p(action) per row
    model_set_input(m, idx, tgt);
    model_forward_bf16(m);
    k_gather_prob(m->a.probs, m->d_tgt, d_curp, B * T, cfg.vocab);
    CK(cudaMemcpy(curp, d_curp, (long)B * T * 4, cudaMemcpyDeviceToHost));
    model_set_input(ref, idx, tgt);
    model_forward_bf16(ref);
    k_gather_prob(ref->a.probs, ref->d_tgt, d_refp, B * T, cfg.vocab);
    CK(cudaMemcpy(refp, d_refp, (long)B * T * 4, cudaMemcpyDeviceToHost));

    // per-token coefficient: mask*(advantage - beta*(1 - refp/curp)) / N
    long Ntok = (long)B * NNEW;                  // completion tokens trained
    for (int row = 0; row < B; row++)
      for (int t = 0; t < T; t++) {
        long i = (long)row * T + t;
        int is_completion = (t >= PLEN - 1 && t <= T - 2);
        if (!is_completion) { h_coef[i] = 0.f; continue; }
        float cp = curp[i] > 1e-8f ? curp[i] : 1e-8f;
        float kl = 1.f - refp[i] / cp;
        h_coef[i] = (adv[row] - beta * kl) / (float)Ntok;
      }
    CK(cudaMemcpy(d_coef, h_coef, (long)B * T * 4, cudaMemcpyHostToDevice));
    model_grpo_backward(m, d_coef);
    model_adamw_step(m, lr, 0.9f, 0.95f, 1e-8f, 0.0f);

    if (step % logevery == 0) {
      double mr = 0; for (int i = 0; i < B; i++) mr += rew[i]; mr /= B;
      float acc = eval_accuracy(e, 0);
      printf("step %4d  mean-reward %.3f  greedy-acc %.3f  | rollout %.0f tok/s, prefix saved %.0f blk/step\n",
             step, mr, acc, roll_tok * 1000.0 / roll_ms, roll_saved / NP);
    }
  }

  printf("final greedy-acc %.3f  (samples:)\n", eval_accuracy(e, 4));
  free(idx); free(tgt); free(adv); free(rew); free(h_coef); free(curp); free(refp); free(ans);
  cudaFree(d_coef); cudaFree(d_curp); cudaFree(d_refp);
  infer_free(e); model_free(m); model_free(ref);
  return 0;
}
