# Milestone 8 — Inference engine (paged KV-cache, continuous batching)

**Window:** Months 9–10 · **Hardware:** 1–4× L40S · **Branch:** `m8-infer`
**Prereq:** M7 (trained weights to serve) · **Read first:** `CONVENTIONS.md`

## Goal
A genuine inference engine in C that generates completions from the trained model — paged KV-cache, continuous batching, sampling. This is the rollout backend the GRPO loop (M9) will drive. Real inference infra, not a toy generate loop. (Conceptually neighbors Lethe, but the cache lives inside this engine; no Lethe integration.)

## Scope
- **Paged KV-cache:** block-table indirection (vLLM-style), fixed-size blocks, allocate/free per sequence. No giant contiguous per-sequence buffers.
- **Continuous batching:** sequences enter/leave the batch at different steps; don't wait for the slowest to finish.
- Sampling: temperature, top-p, greedy. Deterministic mode for testing.
- Reuse the TP path from M4 if serving the full model needs >1 GPU; PP not required for inference.
- Prefix handling: when many sequences share a prompt prefix, compute its KV once and share — note this is where prefix-aware caching would matter (the Lethe-adjacent idea), implemented locally here.

## Oracle (write first)
- Greedy decode vs PyTorch generate on the same weights/prompt: identical token sequence.
- Paged-cache output identical to a contiguous-cache reference.

## Definition of done
1. Oracles pass — paged + continuous batching produces correct tokens.
2. **Rollout throughput measured:** tokens/sec at batch sizes relevant to GRPO group sampling (G=8,16); KV-cache memory efficiency vs naive.
3. Prefix-sharing hit-rate / savings when a group shares a prompt (sets up M9).
4. `PROGRESS.md` updated; merged.

## Notes for Claude Code
- Block size is a tuning knob: small = less waste, more indirection overhead. Measure.
- Continuous batching scheduler bugs show up as wrong tokens for sequences near batch entry/exit — the greedy oracle across staggered entry times catches these.
