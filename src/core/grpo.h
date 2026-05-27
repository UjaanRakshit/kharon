#ifndef KHARON_GRPO_H
#define KHARON_GRPO_H

// GRPO (Group Relative Policy Optimization) helpers. No critic: the advantage for each
// sampled completion is its reward normalized within its group (subtract group mean,
// divide by group std). The policy update then reuses the trainer's backward via
// model_grpo_backward with a per-token coefficient built from these advantages + KL.

#ifdef __cplusplus
extern "C" {
#endif

// Group-relative advantages: adv[g*G + i] = (rewards[g*G+i] - mean_g) / (std_g + eps).
void grpo_advantages(const float *rewards, int n_groups, int group, float eps, float *adv);

#ifdef __cplusplus
}
#endif

#endif
