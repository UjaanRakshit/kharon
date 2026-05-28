# Kharon

A from-scratch **C/CUDA** training + inference + RL stack for a GPT — hand-written kernels, 3D
parallelism, a paged-KV inference engine, and a GRPO RL loop — built to produce one thing: an
**honest, measured performance map** of where bare-metal C beats PyTorch/cuBLAS and where it
loses. No framework underneath; NCCL + cuBLAS are the only libraries.

## Headline results (L40S, measured — see [`bench/REPORT.md`](bench/REPORT.md))

- **BF16 tensor-core GEMM: 3.3×** over fp32 cuBLAS (rel-Frobenius 2.4e-3). The lever the bf16
  mixed-precision path is built on.
- **Single-GPU end-to-end: 0.65× PyTorch eager** (115k vs 176k tok/s, same config) — an honest
  loss: PyTorch's fused tensor-core SDPA beats our hand-written FP32 FlashAttention. The
  identified fix (tensor-core FA) is the top open item.
- **Full 1.2B model on 8× L40S** (TP2×PP2×DP2, ZeRO-1): 30.5k tok/s, **MFU 15.3%**, 19.1 GB/rank.
  The step is **PCIe-comms-bound** (~32% in collectives, no NVLink) — TP all-reduce dominates,
  PP/DP scale better. The interconnect is the ceiling.
- **Paged-KV inference: 2.0–2.67× less memory** than a naive cache; continuous batching scales
  ~16× from G=1→32; prefix sharing saves **1.56 GB at G=32** on the 1.2B model.
- **GRPO RL works:** SFT warm-start → RL refines a pretrained policy to **100%** on a verifiable
  task; cold-start RL provably collapses (documented).

![scaling](bench/plots/scaling.png) ![comms](bench/plots/comms_breakdown.png)

## What's inside (milestones M1–M10)

| | | |
|---|---|---|
| M1 | Single-GPU GPT + AdamW + bit-exact ckpt | PyTorch-oracle verified |
| M2 | Hand-written FlashAttention (fwd+bwd) | vs SDPA |
| M3 | BF16 mixed precision (fp32 master + AdamW) | tensor cores |
| M4–M6 | Tensor + Pipeline (1F1B) parallel, composed | NCCL over PCIe |
| M7 | ZeRO-1 data parallel → full TP×PP×DP 8-GPU mesh | 1.2B |
| M8 | Paged-KV inference engine + continuous batching | rollout backend |
| M9 | GRPO RL loop (no critic) on a verifiable task | closes the loop |
| M10 | Benchmark suite + the honest performance map | the artifact |

Status and every measured number: [`PROGRESS.md`](PROGRESS.md). Toolchain pinned in
[`ENV.md`](ENV.md).

## Build & test

```sh
make all            # kernels, tests, benches, train + grpo_train (nvcc single-driver)
make tp            # multi-GPU targets (NCCL+MPI; cluster)
# Windows dev box: scripts\build.ps1 <make-args>   (sources MSVC vcvars)
```

Correctness is oracle-first — every milestone has a PyTorch-equivalence test before the optimized
code. Run the suite: `./build/test_*.exe` (forward/backward/step/resume/flash/bf16/tp1/pp1/zero/
infer/grpo — all green).

## Reproduce the performance map

```sh
bash bench/run_all.sh        # local microbenchmarks + regenerate all plots
python bench/plot.py         # plots only, from bench/results.json
# cluster series: sbatch scripts/ice_m{6,7,8,9,10}_*.sbatch  (8× L40S for the mesh)
```

All numbers live in [`bench/results.json`](bench/results.json) (single source of truth) and the
plots regenerate from it. The writeup — wins, losses, and the *why* for each — is
[`bench/REPORT.md`](bench/REPORT.md).
