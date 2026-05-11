# PROGRESS.md — running state (update every session)

> Claude Code: read this at session start. Update the active milestone's row and the log at the bottom before ending a session.

## Status board

| # | Milestone | State | Oracle | Benchmark number | Blocker |
|---|-----------|-------|--------|------------------|---------|
| 1 | Single-GPU GPT + ckpt | done | ✓ (4060 + L40S) | 0.64 ms/step 805k tok/s (L40S); 7.1 ms 72k (4060) | — |
| 2 | FlashAttention + fused | done | ✓ FA fwd+bwd vs SDPA | FA 1.5 TFLOP/s, 0.82x cuBLAS (L40S); fusion 1.7x (4060) | beat-cuBLAS needs M3 BF16/TC |
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
- FlashAttention (M2, FP32, L40S, warp-per-row): correct (oracle vs SDPA, fwd+bwd).
  FA fwd 0.73 ms / 1.5 TFLOP/s at T=512 = 0.82x cuBLAS-naive (was 0.19x before opt).
  fwd+bwd 3.0 ms. Model step 0.66 ms (parity with M1's 0.64 ms; no regression from FA).
  Roofline: 1.5 TFLOP/s is ~2% of L40S FP32 peak -> latency/occupancy-bound on the
  per-key warp-shuffle reduction, not compute- or bandwidth-bound. Closing to >1x needs
  a blocked-matmul formulation or (the real lever) BF16 tensor cores in M3.
- Fused elementwise (M2): 4060 1.36-1.71x; L40S 0.96x (48MB L2 absorbs the intermediate).
- FlashAttention vs official: __ (deferred to M3 — official FA is BF16/tensor-core)
- Custom GEMM vs cuBLAS (shape __): __
- Bubble fraction @ m=__ microbatches: __
- TP=2 comms/compute overlap (L40S PCIe): __
- A100-NVLink vs L40S-PCIe TP=2: __
- Full 8-GPU mesh tokens/sec & MFU: __
- vs Megatron-LM same config: __
- GRPO reward curve delta: __

## Session log
<!-- newest first: date — what was done — what's next -->
- 2026-05-29 — M2 done: hand-written FlashAttention (fwd+bwd, tiled online softmax,
  causal) oracle-verified vs PyTorch SDPA; swapped into the model (M1 oracles still pass).
  Optimized FP32 FA to warp-per-row: FA fwd 0.4->1.5 TFLOP/s, 0.19x->0.82x cuBLAS, model
  step 1.10->0.66 ms (parity with M1). Fused bias+residual / bias+gelu (4060 1.3-1.7x,
  L40S neutral). Validated on L40S (job 5347604). Next: M3 BF16 + tensor cores (FA win).
- 2026-05-29 — M1 validated on PACE ICE L40S (job 5347432, node atl1-1-03-004-27):
  all four oracles PASS, benchmark 0.64 ms/step / 805k tok/s (vs 7.1 ms / 72k on 4060).
  M1 DoD fully met. SSH key access + cluster env set up (see ENV.md). Next: M2.
- 2026-05-29 — M1 complete on 4060: C/CUDA GPT fwd+bwd, AdamW, checkpoint/resume.
  All four oracles pass (forward logits+loss, backward grads, AdamW post-step params,
  bit-exact resume) at FP32 tol. Baseline 7.1 ms/step. Build: nvcc+MSVC via
  scripts/build.ps1 (Windows) / make (cluster). Next: M2 FlashAttention, and run M1
  on one L40S for the cluster number once on ICE.
- (start)
