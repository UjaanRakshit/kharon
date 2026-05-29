# Kharon

A from-scratch C/CUDA stack that trains, serves, and RL-tunes a GPT with hand-written kernels and 3D parallelism. It trains a 1.2B-parameter model across 8x L40S, serves it through a paged-KV inference engine, and closes the loop with a GRPO reinforcement-learning trainer. No deep-learning framework underneath: NCCL and cuBLAS are the only dependencies.

[Quickstart](#quickstart) | [What's Inside](#whats-inside) | [Architecture](#architecture) | [Results](#results) | [Correctness](#correctness) | [Build](#build-test-install)

## Highlights

- **Trains a 1.2B GPT on 8x L40S** with three composed axes of parallelism (tensor x pipeline x ZeRO-1 data), at 15.3% MFU with a full per-axis compute/communication profile.
- **Hand-written CUDA kernels:** bf16 tensor-core GEMM at **3.3x** over fp32 cuBLAS, FlashAttention (forward + backward), fused bias+residual / bias+gelu, LayerNorm / softmax / cross-entropy, all templated on storage dtype.
- **BF16 mixed precision** done right: fp32 master weights + AdamW, bf16 compute, fp32 reduction stats.
- **Paged-KV inference engine** (vLLM-style block tables, continuous batching, prefix sharing): **2.0-2.67x** less KV memory than a naive cache and ~16x throughput scaling from batch 1 to 32.
- **GRPO RL loop** (no critic): group-relative advantages + KL control, reaching **100%** on a verifiable task.
- **Every component verified against PyTorch** (bit-exact where applicable) with a one-command, reproducible benchmark suite.

Hardware: NVIDIA L40S (Ada sm_89, 48 GB, PCIe), CUDA 12.6.1, NCCL nvhpc-24.5, OpenMPI 4.1.8. Dev box: RTX 4060 Laptop. All numbers below are measured, not projected.

## Quickstart

```sh
make all
for t in build/test_*.exe; do "$t"; done      # 13 PyTorch-equivalence oracles
```

Single-GPU bf16 training with bit-exact checkpoint/resume:

```sh
python scripts/prep_data.py data/input.bin
./build/train.exe --layers 8 --d 512 --heads 8 --seq 256 --batch 32 \
    --steps 1000 --lr 3e-4 --data data/input.bin --ckpt checkpoints/train.ckpt
```

Full 3D mesh (tensor x pipeline x ZeRO-1 data parallel) on 8 GPUs:

```sh
mpirun -n 8 ./build/pp_train.exe --tp 2 --dp 2 \
    --layers 24 --d 2048 --heads 16 --seq 512 --mbatch 4 --M 8 --steps 200 --data data/input.bin
```

Paged-KV inference throughput, and the GRPO RL loop:

```sh
./build/bench_infer.exe --layers 24 --d 1024 --heads 16 --vocab 50257 --seq 1024 --prompt 128 --new 256
./build/grpo_train.exe --layers 6 --d 512 --heads 8 --G 16 --prompts 32 --sft 200 --steps 400
```

## What's Inside

| Layer | Implementation |
|---|---|
| Model | Pre-norm GPT (GELU, weight-tied head), AdamW (decoupled decay), bit-exact checkpoint/resume |
| Kernels | cuBLAS bf16 tensor-core GEMMs, hand-written FlashAttention (fwd+bwd), fused elementwise, LayerNorm / softmax / cross-entropy |
| Precision | BF16 mixed precision: fp32 master + AdamW, bf16 compute, fp32 reduction stats |
| Parallelism | Tensor (Megatron column/row), Pipeline (1F1B, deadlock-free batched send/recv), ZeRO-1 data parallel, composed as TP x PP x DP |
| Inference | Paged KV-cache, continuous batching, prefix sharing (copy-on-write), temperature/greedy sampling |
| RL | GRPO (no critic): group-relative advantages, KL to a frozen reference, policy gradient through the trainer |

Built as ten milestones, each with a PyTorch-equivalence oracle written before the optimized kernel: single-GPU GPT (M1), FlashAttention (M2), bf16 + tensor cores (M3), tensor / pipeline / composed parallelism (M4-M6), ZeRO-1 + the full 8-GPU mesh (M7), the inference engine (M8), GRPO (M9), and the benchmark suite (M10).

## Architecture

Three layers: hand-written CUDA kernels in `src/kernels/` and `src/core/`, a NCCL+MPI communication layer in `src/parallel/`, and thin C entrypoints that wire them into trainers and engines. PyTorch is never a runtime dependency; it appears only as the oracle that test fixtures are generated against.

```
src/core/      gpt.c: forward/backward (fp32 + bf16 + TP + PP variants), AdamW, GRPO backward.
               kernels.cu: GEMM wrappers (cublasGemmEx), LayerNorm/GELU/softmax/cross-entropy,
               casts, templated on storage dtype. infer.c: paged-KV engine. grpo.c: advantages.
               ckpt.c, adamw.c, arena.c (bump allocator), data.c, refio.c.

src/kernels/   flash.cu: FlashAttention (warp-per-row online softmax, causal, fwd+bwd).
               paged.cu: inference kernels (per-token embed, append-KV, paged attention, gather).

src/parallel/  comms.cu: NCCL wrapper. 3D grid via ncclCommSplit (TP + DP sub-comms, PP p2p on
               the global comm); bf16 all-reduce, batched send/recv, fp32 reduce-scatter/all-gather.

src/           entrypoints: train.c, pp_train.c (3D mesh + 1F1B scheduler), grpo_train.c.
tests/         13 oracles + PyTorch reference generators.   bench/  results + plots + harness.
scripts/       build wrapper + Slurm jobs for the cluster runs.
```

The build is a single `nvcc` driver (`-arch=sm_89`); the multi-GPU targets add `-ccbin mpicxx` and NCCL.

## Results

**Parallel scaling (2-4x L40S, d512x8L proxy).** Pipeline parallelism scales near-linearly; the composed TP x PP mesh reaches 2.35x on 4 GPUs.

![parallel scaling](bench/plots/scaling.png)

**Full 1.2B model on 8 GPUs** (TP2 x PP2 x DP2, ZeRO-1): 30.5k tok/s, **MFU 15.3%**, 19.1 GB/rank (the full model fits with room because ZeRO-1 shards the Adam moments to 1/DP). The per-step breakdown isolates compute from each communication axis:

![8-GPU step breakdown](bench/plots/comms_breakdown.png)

| stage | compute | TP all-reduce | PP bubble | DP optimizer-comm |
|---|---:|---:|---:|---:|
| ms / step | 729 | 130 | 111 | 102 |

**Inference (paged-KV).** Continuous batching scales rollout throughput ~16x from G=1 to G=32; the paged cache uses 2.0-2.67x less memory than reserving the full sequence, and prefix sharing saves up to 1.56 GB at G=32 on the 1.2B model.

![rollout throughput](bench/plots/inference.png)

| group size G | 1 | 8 | 16 | 32 |
|---|---:|---:|---:|---:|
| 350M tok/s | 166 | 810 | 1,619 | 2,693 |
| 1.2B tok/s | 82 | 531 | 941 | 1,606 |

**GRPO reinforcement learning.** An SFT warm-start brings a from-scratch policy to ~90% on a verifiable task; GRPO then refines it to **100%**, with rollouts served by the inference engine (prefix-shared, ~51k tok/s).

![GRPO reward curve](bench/plots/grpo.png)

The benchmark suite regenerates every number and plot from one entrypoint (`bash bench/run_all.sh`); all numbers live in [`bench/results.json`](bench/results.json).

## Correctness

Every kernel is checked against a deterministic reference before it ships: the fp32 path is verified against PyTorch, the bf16 path against the fp32 path, parallel runs against the single-GPU loss curve, and decoded tokens against PyTorch greedy token-for-token. Thirteen oracles run on a clean build.

| Component | Reference | Result |
|---|---|---|
| Forward / backward / AdamW step | PyTorch fp32 | passes |
| FlashAttention (fwd+bwd) | PyTorch SDPA | passes |
| BF16 forward + grads | fp32 path | within bf16 tol |
| Tensor / pipeline parallel | single-GPU loss curve | matches |
| ZeRO-1 update | unsharded AdamW | bit-identical |
| Checkpoint/resume (incl. 8-GPU sharded mesh) | continuous run | bit-exact |
| Paged decode / paged-vs-contiguous | PyTorch greedy | token-for-token |
| GRPO advantages + policy-gradient backward | reference formula / CE backward | exact |

## Build, Test, Install

Requires the CUDA toolkit 12.x and a C11 host compiler; multi-GPU targets add MPI and NCCL (cluster modules: `cuda/12.6.1 gcc/12.3.0 nvhpc-nccl/24.5 openmpi/4.1.8-cuda`).

```sh
make all            # kernels, 13 tests, benches, train + grpo_train (single nvcc driver)
make tp             # multi-GPU targets (NCCL + MPI)
for t in build/test_*.exe; do "$t"; done
```

On the Windows dev box, `scripts\build.ps1 <make-args>` sources the MSVC environment. Toolchain pinned in [`ENV.md`](ENV.md).
