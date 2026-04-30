# Milestone 10 — Benchmark suite + performance writeup (the headline artifact)

**Window:** Months 11–12 · **Hardware:** L40S (primary), A100×2 (comparison), 8×H100 (stretch) · **Branch:** `m10-bench`
**Prereq:** M1–M9 · **Read first:** `CONVENTIONS.md`

## Goal
Produce the rigorous, honest performance map that is the project's headline deliverable: where hand-written C beats PyTorch/Megatron, by how much, and why — and where it loses. This is the data-backed answer to unsubstantiated "order of magnitude vs JAX" claims, and the strongest interview artifact.

## Scope
- **Benchmark harness** (`bench/`): reproducible, scripted, one command regenerates every number and plot.
- Baselines, same model/config/hardware:
  - Your FlashAttention vs official FlashAttention.
  - Your custom GEMM (if built) vs cuBLAS, on your layer shapes.
  - Your 3D-parallel training vs **Megatron-LM** (and/or a PyTorch FSDP/pipeline baseline).
  - Your inference engine vs vLLM on rollout throughput.
- **Metrics:** tokens/sec, MFU, bubble fraction vs microbatch count, comms/compute overlap %, memory high-water per rank, per-kernel % of vendor.
- **Interconnect study:** L40S-PCIe vs A100-NVLink TP=2 (from M4 stretch) — quantify what NVLink buys for tensor parallelism.
- **Stretch headline:** one 8×H100 run for a top-end number (queue permitting).

## Definition of done
1. `bench/` regenerates all numbers + plots from one entrypoint.
2. **Writeup** (`bench/REPORT.md`): the performance map — wins, losses, and the *why* for each, with plots. Honest about where frameworks win.
3. All plots checked into the repo.
4. `PROGRESS.md` "key measured numbers" fully populated.
5. README updated with the headline results and how to reproduce.

## Notes for Claude Code
- Honesty is the value here. "Megatron is 1.4× faster on the full mesh but our FA is 0.95× official and our GEMM beats cuBLAS by 12% on shape X" is a *stronger* artifact than an inflated single number. Recruiters in AI-infra can tell the difference.
- Pin every version in `ENV.md` and stamp results with it — unversioned benchmarks are worthless.
- The interconnect study (PCIe vs NVLink) is genuinely novel-feeling content for a student project; give it its own section.
