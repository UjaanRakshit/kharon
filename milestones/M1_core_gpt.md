# Milestone 1 — Single-GPU correct C/CUDA GPT (fwd+bwd) + checkpoint/resume

**Window:** Months 1–2 · **Hardware:** RTX 4060 (dev), one L40S (validate) · **Branch:** `m1-core`
**Prereq:** none · **Read first:** `CONVENTIONS.md`

## Goal
A from-scratch GPT (forward + backward) in C/CUDA whose logits and loss match a reference PyTorch GPT to tolerance, with a memory arena you own and bit-exact checkpoint/resume. No parallelism, no custom kernels yet — cuBLAS for GEMM, a straightforward attention is fine. This is the correctness foundation everything else builds on; get it right where iteration is cheap (4060).

## Scope
- Model config struct: layers, d_model, heads, vocab, seq, dropout(off for determinism).
- Forward: embedding + positional, N transformer blocks (LN, QKV via cuBLAS, attention, MLP, residuals), final LN, LM head.
- Backward: hand-written autograd glue (manual chain through the same ops; no framework). cuBLAS for matmul grads.
- AdamW optimizer, FP32 throughout for this milestone (BF16 deferred to M3).
- **Memory arena** (`src/core/`): own slab allocator, no `cudaMalloc` per-op. Track high-water by buffer class.
- **Checkpoint/resume:** serialize weights+optimizer+RNG+step; resume must reproduce the uninterrupted loss curve bit-exact.

## Oracle (write first, in `tests/`)
- Build the same-config GPT in PyTorch with fixed seed and identical init; dump weights to a file.
- C stack loads those weights, runs forward on a fixed batch → compare logits (`rtol=1e-4, atol=1e-5`).
- One backward step → compare grads per tensor at same tolerance.
- Checkpoint→resume→one more step → loss equals the no-interrupt path.

## Definition of done
1. Logits + grads match PyTorch at FP32 tolerance.
2. Checkpoint/resume bit-exact.
3. Forward+backward step time recorded on 4060 and on one L40S (first real number).
4. `PROGRESS.md` updated; `m1-core` merged.

## Notes for Claude Code
- Build per-arch: 4060 is `sm_89`? No — 4060 is Ada `sm_89` like L40S actually (Ada Lovelace). Confirm with `nvcc --gpu-architecture` and `nvidia-smi`; set arch flag accordingly in the Makefile.
- Keep the autograd glue dead simple and explicit — readability now saves debugging in M3–M7.
- Determinism: fixed seeds, dropout off, deterministic cuBLAS algo where available; this is what makes the oracle meaningful.
