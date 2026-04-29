# PROGRESS.md — running state (update every session)

> Claude Code: read this at session start. Update the active milestone's row and the log at the bottom before ending a session.

## Status board

| # | Milestone | State | Oracle | Benchmark number | Blocker |
|---|-----------|-------|--------|------------------|---------|
| 1 | Single-GPU GPT + ckpt | not started | — | — | — |
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
- CUDA: __  NCCL: __  Driver: __
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
- (start)
