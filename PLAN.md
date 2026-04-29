# Kharon — Bare-Metal C Distributed Training + RL Stack

**Duration:** ~12 months, solo, driven alongside Claude Code
**Primary hardware:** PACE **ICE** cluster, `ice-gpu` partition — **8× L40S per node** (48GB, Ada, PCIe Gen4, **no NVLink**), 4 such nodes, 32 L40S total. Local dev on RTX 4060.
**Walltime cap:** 16h per job (`MaxTime=16:00:00`) → **checkpoint/resume is mandatory, milestone 1 onward.**
**Comparison hardware:** A100×2/node (NVLink, for a TP interconnect comparison); H100×8/node (stretch benchmark only — heavily contested).

**One-line:** A from-scratch C/CUDA system that trains a ~1B-param transformer with 3D parallelism on a single 8×L40S node, using hand-written kernels and a co-located GRPO RL engine, benchmarked rigorously against PyTorch/Megatron to map exactly where bare-metal wins and where it loses.

---

## Thesis

Own every layer JAX/XLA and Megatron abstract away — kernels, memory, parallelism, comms, scheduler, RL loop — and **measure precisely where hand-written C beats them and where it loses**. The honest performance map is the headline deliverable, and the data-backed reply to unbacked "order of magnitude vs JAX" claims.

Two hardware truths drive every decision:
- **No NVLink on L40S** → TP collectives go over PCIe Gen4. Numbers reflect this; document it as a real constraint.
- **16h walltime** → everything checkpoints and resumes. Train in resumable segments, never one long run.

## Confirmed mesh

8 GPUs in one L40S node → **TP=2 × PP=2 × DP=2**, entirely intra-node over PCIe. No InfiniBand hop. The 8 GPUs span 4 socket/NUMA affinity domains (`S:0-3`) — pin ranks to the correct CPU socket for the PCIe path (see M3 brief).

## Model

~1B params: 24 layers, d_model 2048, 16 heads, vocab ~50k, seq 1024 (dev at 256). Big enough that all three parallel axes matter; small enough to stay iteration-bound and fit resumable 16h segments.

## Scope

**In:** hand-written kernels competitive on your shapes; 3D parallelism hand-implemented; real inference engine (paged KV, continuous batching) driving GRPO; rigorous benchmark suite.
**Out:** MoE/expert parallelism; multi-node (single node has all 8 GPUs); learned reward model (verifiable rewards only); Lethe integration (separate project).

## Milestones

| # | Title | Window | Done when |
|---|-------|--------|-----------|
| 1 | Single-GPU correct GPT (fwd+bwd) + checkpoint/resume | M1–2 | Logits+loss match PyTorch to tol on 4060; resume bit-exact |
| 2 | Hand-written FlashAttention + fused blocks | M2–3 | Correct + benchmarked vs official kernel |
| 3 | BF16 mixed precision + full training step | M3–4 | Trains to sane loss on small corpus |
| 4 | Tensor parallel (TP=2) on L40S | M4–5 | Matches single-GPU loss; PCIe TP comms measured |
| 5 | Pipeline parallel (1F1B) | M5–7 | Bubble fraction measured + plotted |
| 6 | Compose TP×PP (4 GPU) | M7–8 | Stable, loss matches single-GPU |
| 7 | ZeRO-1 DP → full TP×PP×DP (8 GPU) | M8–9 | 8-GPU mesh stable; mem/rank reported |
| 8 | Inference engine (paged KV, cont. batching) | M9–10 | Rollout throughput measured |
| 9 | GRPO loop on verifiable task | M10–11 | Reward curve moves |
| 10 | Benchmark suite + writeup | M11–12 | Performance map published |

**Stretch (slack absorbers):** custom GEMM beating cuBLAS on your shapes; ZeRO-2; interleaved 1F1B; A100-NVLink-vs-L40S-PCIe TP comparison; one 8×H100 headline run.

## How to use this repo with Claude Code

- Keep `PLAN.md` (this file), `CONVENTIONS.md`, and `PROGRESS.md` in the root. Point the agent at all three at session start.
- Each milestone has a self-contained brief in `milestones/`. Feed the agent **one milestone brief at a time** — they're written to be the unit of work.
- Update `PROGRESS.md` at the end of every milestone (done / measured / blocker). Agentic sessions degrade without a running state file.
- **Correctness-test-first:** every kernel and parallel mode gets its PyTorch-equivalence oracle written before the optimized code. Highest-leverage habit with an agent.
- Make the agent **measure, not assert**: "done" always includes the benchmark number vs the vendor/framework baseline.
- 4060 = inner loop (correctness, kernel iteration, cheap). ICE L40S = outer loop (multi-GPU validation, final benchmarks). Spend allocation deliberately.

## Open items to confirm on-cluster (non-blocking)

- `nvidia-smi topo -m` inside an L40S job → confirm PCIe topology label (PIX/PHB/SYS) and which GPU pairs share a switch (affects TP rank placement).
- CUDA / NCCL module versions on RHEL9 ICE → pin in `ENV.md`.
