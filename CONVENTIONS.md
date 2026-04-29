# CONVENTIONS.md — shared rules for all milestones

Every milestone brief assumes these. Claude Code should read this file plus the one active milestone brief.

## Language & build
- **C (C11)** for host/runtime/driver. CUDA C for kernels. A *little* C++ is tolerated only where CUDA requires it (templated kernels, `__device__` lambdas) — keep it minimal and isolated.
- Build with `make` + `nvcc`. No CMake unless a milestone outgrows make. One `Makefile`, debug/release flags.
- Pin toolchain in `ENV.md`: CUDA version, NCCL version, driver, GPU arch flag (`-arch=sm_89` for L40S Ada; `sm_75` for the 4060 is Turing → `sm_89`/`sm_86` mismatch, so build per-target).

## Repo layout
```
kharon/
  PLAN.md  CONVENTIONS.md  PROGRESS.md  ENV.md
  milestones/        # the briefs
  src/
    core/            # M1: tensors, memory arena, autograd glue
    kernels/         # M2: attention, gemm, fused blocks
    parallel/        # M3-7: tp, pp, dp, comms
    infer/           # M8: kv cache, batching, sampling
    rl/              # M9: grpo loop, rewards
  tests/             # pytorch-equivalence oracles per module
  bench/             # M10: benchmark harness + plots
  scripts/           # slurm submit scripts, checkpoint tools
  third_party/       # pytorch baselines, reference kernels
```

## Correctness-test-first (non-negotiable)
- For every kernel and every parallel mode, write the **PyTorch-equivalence oracle in `tests/` BEFORE** the optimized implementation.
- Tolerance: FP32 `rtol=1e-4, atol=1e-5`; BF16 `rtol=2e-2, atol=2e-2` (looser, document why).
- A milestone is not "done" until its oracle passes AND its benchmark number exists.

## Measure, don't assert
- Every kernel ships a microbenchmark vs the vendor/official kernel (cuBLAS, official FlashAttention).
- Every parallel mode ships: tokens/sec, and where relevant bubble fraction, comms/compute overlap %, memory high-water per rank.
- Numbers go in `PROGRESS.md` and later `bench/`. "It works" never substitutes for "X% of cuBLAS on shape Y."

## Checkpoint/resume (mandatory from M1)
- 16h walltime cap → every training entrypoint must checkpoint optimizer+weights+RNG+step and **resume bit-exact** (verify by checkpoint→resume→compare loss to uninterrupted run).
- Checkpoint cadence: time-based (every ~20 min) not just step-based, so a job killed at 16h loses little.
- Slurm scripts in `scripts/` request `--time=16:00:00`, `--gres=gpu:l40s:N`, and auto-resubmit on requeue.

## Hardware facts to code against
- L40S: 48GB, Ada `sm_89`, BF16-strong, **no FP64 to speak of**, PCIe Gen4, **no NVLink**.
- 8 GPUs/node, 4 socket domains (`S:0-3`). Pin ranks to sockets; check `nvidia-smi topo -m` for which pairs share a PCIe switch and place TP pairs on the closest pair.
- Mesh: TP=2 × PP=2 × DP=2, intra-node.

## Numerics
- BF16 compute, FP32 master weights, FP32 optimizer state.
- Loss-scaling not needed for BF16 (unlike FP16) — note this rather than copying FP16 recipes.

## Definition of done (every milestone)
1. Oracle test passes at stated tolerance.
2. Benchmark number recorded vs baseline.
3. Checkpoint/resume still bit-exact (for training-path milestones).
4. `PROGRESS.md` updated.
5. Merged to main, branch green.
