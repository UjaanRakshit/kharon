#include "model.h"
#include "kernels.h"
#include "kharon.h"
#include <math.h>
#include <cuda_runtime.h>

// One AdamW step over the flat param arena. Weights, grads and moments share the
// same arena layout, so a single kernel sweep covers every parameter.
void model_adamw_step(Model *m, float lr, float b1, float b2, float eps, float wd) {
  m->step++;
  long n = m->w_arena.off / 4;
  float bc1 = 1.f - powf(b1, (float)m->step);
  float bc2 = 1.f - powf(b2, (float)m->step);
  k_adamw((float *)m->w_arena.base, (const float *)m->g_arena.base,
          m->opt_m, m->opt_v, n, lr, b1, b2, eps, wd, bc1, bc2);
}

// ZeRO-1: re-size the Adam moment buffers to this rank's 1/DP shard and allocate a
// reduce-scatter landing buffer. Params are padded up to a multiple of dp_size so the
// scatter divides evenly; the tail padding is zeroed once so it never perturbs Adam.
void model_enable_zero(Model *m, int dp_size, int dp_rank) {
  m->dp_size = dp_size;
  m->dp_rank = dp_rank;
  long n = m->w_arena.off / 4;
  long npad = ((n + dp_size - 1) / dp_size) * dp_size;
  long shard = npad / dp_size;
  m->zero_npad = npad;
  m->zero_n = shard;
  m->zero_off = (long)dp_rank * shard;
  if (npad > n) {                                  // zero the grad/param padding tail
    CK(cudaMemset(m->g_arena.base + n * 4, 0, (npad - n) * 4));
    CK(cudaMemset(m->w_arena.base + n * 4, 0, (npad - n) * 4));
  }
  arena_destroy(&m->om_arena); arena_destroy(&m->ov_arena);
  long sb = shard * 4;
  m->om_arena = arena_create("adam_m", sb, 1);
  m->ov_arena = arena_create("adam_v", sb, 1);
  m->opt_m = (float *)arena_alloc(&m->om_arena, sb);
  m->opt_v = (float *)arena_alloc(&m->ov_arena, sb);
  CK(cudaMemset(m->opt_m, 0, sb));
  CK(cudaMemset(m->opt_v, 0, sb));
  CK(cudaMalloc((void **)&m->grad_shard, sb));
}

// ZeRO-1 step: reduce-scatter the full fp32 grads into this rank's shard (DP-averaged),
// run AdamW on the matching slice of the fp32 master + sharded moments, then all-gather
// the updated master slices back into the full param arena for the next forward.
void model_adamw_step_zero(Model *m, float lr, float b1, float b2, float eps, float wd) {
  m->step++;
  float bc1 = 1.f - powf(b1, (float)m->step);
  float bc2 = 1.f - powf(b2, (float)m->step);
  long off = m->zero_off, shard = m->zero_n;
  m->dp_reduce_scatter(m->dp_ctx, (const float *)m->g_arena.base, m->grad_shard, shard);
  k_adamw((float *)m->w_arena.base + off, m->grad_shard, m->opt_m, m->opt_v, shard,
          lr, b1, b2, eps, wd, bc1, bc2);
  m->dp_all_gather(m->dp_ctx, (const float *)m->w_arena.base + off,
                   (float *)m->w_arena.base, shard);
}
