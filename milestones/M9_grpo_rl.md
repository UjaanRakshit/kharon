# Milestone 9 — GRPO RL loop on a verifiable task

**Window:** Months 10–11 · **Hardware:** 4–8× L40S · **Branch:** `m9-grpo`
**Prereq:** M7 (trainer), M8 (rollout engine) · **Read first:** `CONVENTIONS.md`

## Goal
Close the loop: GRPO (Group Relative Policy Optimization) using the M8 inference engine for rollouts and the M7 trainer for updates. GRPO fits the "pure inference stack" framing because it has **no critic** — advantages come from group-relative reward normalization, so the inference engine is the whole rollout path.

## Scope
- **Group sampling:** for each prompt, sample G completions (G=8 or 16) via M8 — shared prompt prefix means M8's prefix-sharing pays off directly.
- **Verifiable reward:** start with arithmetic or strict-format tasks (reward = correctness, programmatically checkable). Stretch: code-execution reward. No learned reward model.
- **Group-relative advantage:** normalize rewards within the group (subtract group mean, divide by group std); no value function.
- Policy update through the M7 mesh; KL control to a reference policy as standard.
- Train the pretrained 1B policy on the verifiable task; log reward over steps.

## Oracle (write first)
- GRPO advantage computation vs a reference implementation (small, CPU, numpy) on toy reward arrays: identical.
- One policy-update step vs a PyTorch GRPO reference on the same rollouts/seed: gradients match at BF16 tolerance.

## Definition of done
1. Oracles pass.
2. **Reward curve moves:** held-out task accuracy / mean reward increases over training. This is the deliverable — RL that demonstrably works.
3. Rollout efficiency: tokens/sec during generation, prefix-reuse savings across the group (the M8→M9 payoff).
4. `PROGRESS.md` updated; merged.

## Notes for Claude Code
- Reward hacking shows up fast on verifiable tasks (model finds a degenerate output that scores) — log sample completions, not just scalar reward.
- Keep the reference policy frozen for KL; a drifting reference is a silent bug.
- The generate↔train alternation must share weights cleanly — after each update, the M8 engine serves the new policy. Define that handoff explicitly.
