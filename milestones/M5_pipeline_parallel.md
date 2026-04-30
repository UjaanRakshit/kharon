# Milestone 5 — Pipeline parallelism (1F1B)

**Window:** Months 5–7 · **Hardware:** 2–4× L40S (one node) · **Branch:** `m5-pp`
**Prereq:** M4 · **Read first:** `CONVENTIONS.md`

## Goal
Split the model into pipeline stages across GPUs and run a 1F1B (one-forward-one-backward) schedule, with inter-stage activation/gradient transfer over NCCL point-to-point. The bubble fraction is the headline metric.

## Scope
- Stage partition: divide the 24 layers into P stages (start P=2, then P=4). Balance params/compute per stage.
- **1F1B steady-state scheduler:** warmup (fill), steady 1F1B, cooldown (drain). Bound activation memory to ~P microbatches, not all M.
- Inter-stage transfer: `ncclSend`/`ncclRecv` of activations forward, gradients backward. Overlap transfer with compute of adjacent microbatches.
- Microbatch count M as a tunable; sweep it.

## Oracle (write first)
- PP forward+backward vs single-GPU M3 path: same loss curve at BF16 tolerance (pipelining must not change the math, only the schedule).

## Definition of done
1. Oracle passes.
2. **Bubble fraction measured and plotted vs M:** verify it tracks (P−1)/(M+P−1); show driving it down by raising M (e.g. P=4: M=16→~16%, M=32→~8.8%).
3. Comms/compute overlap % per stage recorded.
4. Step-time breakdown: compute / comms / bubble / optimizer.
5. `PROGRESS.md` updated; merged.

## Stretch
- **Interleaved 1F1B** (multiple stage-chunks per GPU) for lower bubble — the Megatron interleaved variant. Compare bubble vs vanilla.

## Notes for Claude Code
- The scheduler is the deliverable, not the kernels. Get warmup/steady/cooldown microbatch bookkeeping exactly right — off-by-one here corrupts grads silently; the oracle catches it.
- Activation stashing for backward is the memory subtlety: stash only what 1F1B requires, free aggressively.
