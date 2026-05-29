#include "grpo.h"
#include <math.h>

// Group-relative advantage: within each group of `group` completions, center by the
// group mean and scale by the group std. This is the whole "no critic" trick - the
// baseline is the group itself, so reward variance inside a group drives the update.
void grpo_advantages(const float *rewards, int n_groups, int group, float eps, float *adv) {
  for (int g = 0; g < n_groups; g++) {
    const float *r = rewards + (long)g * group;
    double mean = 0;
    for (int i = 0; i < group; i++) mean += r[i];
    mean /= group;
    double var = 0;
    for (int i = 0; i < group; i++) { double d = r[i] - mean; var += d * d; }
    var /= group;
    double std = sqrt(var);
    float *a = adv + (long)g * group;
    for (int i = 0; i < group; i++) a[i] = (float)((r[i] - mean) / (std + eps));
  }
}
