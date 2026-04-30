# Milestone 2 — Hand-written FlashAttention + fused transformer blocks

**Window:** Months 2–3 · **Hardware:** RTX 4060 (dev), one L40S (benchmark) · **Branch:** `m2-kernels`
**Prereq:** M1 (correct GPT to swap kernels into) · **Read first:** `CONVENTIONS.md`

## Goal
Replace the naive attention from M1 with a hand-rolled FlashAttention kernel, and fuse the cheap elementwise ops around the transformer block. This is the canonical "can write a hard kernel" proof and leverages your Pyre CUDA experience.

## Scope
- **FlashAttention (forward + backward):** tiled, online softmax, causal mask, BF16-ready (FP32 accum). The backward pass is the hard part — budget time for it.
- **Fused block kernel(s):** fuse LayerNorm + bias-add + residual, and the MLP activation (GELU) + bias, where profiling shows it pays. Measure fused vs unfused.
- Keep cuBLAS for the big GEMMs (QKV proj, MLP) this milestone — custom GEMM is a stretch goal, not here.
- Each kernel: occupancy + roofline note (is it memory- or compute-bound on L40S?).

## Oracle (write first)
- FlashAttention output + grads vs PyTorch `scaled_dot_product_attention` (or official FA) at BF16 tolerance.
- Fused block vs the M1 unfused path: same logits/grads at tolerance.

## Definition of done
1. Oracles pass.
2. **FlashAttention microbenchmark vs official FlashAttention** on L40S — record TFLOP/s and % of official, at your seq len and head dim.
3. Fused-vs-unfused block speedup recorded.
4. M1 oracle still passes with kernels swapped in (no correctness regression).
5. `PROGRESS.md` updated; merged.

## Notes for Claude Code
- Tile sizes are L40S-specific (96KB shared mem/SM on Ada). Don't copy A100/H100 tile configs blindly; tune for `sm_89`.
- Validate forward before attempting backward — FA backward bugs are brutal to debug without a green forward.
- L40S has weak FP64; keep everything FP32-accum / BF16-storage, never FP64.
