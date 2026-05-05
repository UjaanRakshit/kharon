#include "model.h"
#include "kernels.h"
#include <math.h>

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
