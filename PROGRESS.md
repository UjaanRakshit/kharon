# PROGRESS.md — running state (update every session)

> Claude Code: read this at session start. Update the active milestone's row and the log at the bottom before ending a session.

## Status board

| # | Milestone | State | Oracle | Benchmark number | Blocker |
|---|-----------|-------|--------|------------------|---------|
| 1 | Single-GPU GPT + ckpt | benchmarked | ✓ fwd+bwd+step | 7.1 ms/step, 71.9k tok/s (4060) | L40S number pending cluster |
| 2 | FlashAttention + fused | not started | — | — | — |
| 3 | BF16 + full step | not started | — | — | — |
| 4 | TP=2 | not started | — | — | — |
| 5 | PP (1F1B) | not started | — | — | — |
| 6 | TP×PP | not started | — | — | — |
| 7 | +ZeRO-1 DP (8 GPU) | not started | — | — | — |
| 8 | Inference engine | not started | — | — | — |
| 9 | GRPO loop | not started | — | — | — |
| 10 | Benchmark + writeup | not started | — | — | — |

States: not started / in progress / oracle-passing / benchmarked / done

## Environment (fill from cluster)
- Dev box (confirmed): RTX 4060 Laptop, sm_89, 8GB, CUDA 12.6, driver 596.49, MSVC 14.41. See ENV.md.
- CUDA: __  NCCL: __  Driver: __  (cluster)
- L40S topo (`nvidia-smi topo -m`): __ (which pairs share a PCIe switch?)
- Module load lines: __

## Key measured numbers (the resume material)
- FlashAttention vs official: __
- Custom GEMM vs cuBLAS (shape __): __
- Bubble fraction @ m=__ microbatches: __
- TP=2 comms/compute overlap (L40S PCIe): __
- A100-NVLink vs L40S-PCIe TP=2: __
- Full 8-GPU mesh tokens/sec & MFU: __
- vs Megatron-LM same config: __
- GRPO reward curve delta: __

## Session log
<!-- newest first: date — what was done — what's next -->
- 2026-05-29 — M1 complete on 4060: C/CUDA GPT fwd+bwd, AdamW, checkpoint/resume.
  All four oracles pass (forward logits+loss, backward grads, AdamW post-step params,
  bit-exact resume) at FP32 tol. Baseline 7.1 ms/step. Build: nvcc+MSVC via
  scripts/build.ps1 (Windows) / make (cluster). Next: M2 FlashAttention, and run M1
  on one L40S for the cluster number once on ICE.
- (start)
