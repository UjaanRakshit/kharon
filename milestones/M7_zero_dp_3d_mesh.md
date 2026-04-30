# Milestone 7 — ZeRO-1 data parallel → full TP×PP×DP mesh (8 GPU)

**Window:** Months 8–9 · **Hardware:** 8× L40S (full node) · **Branch:** `m7-3d`
**Prereq:** M6 · **Read first:** `CONVENTIONS.md`

## Goal
Add data parallelism with ZeRO-1 optimizer-state sharding and compose all three axes: **TP=2 × PP=2 × DP=2 = 8 GPUs, one node, all PCIe**. This is the full ~1B model training, and the project's structural centerpiece.

## Scope
- **Data parallel** across DP=2 replicas of the TP×PP mesh: all-reduce gradients across DP ranks.
- **ZeRO-1:** shard optimizer state (FP32 master + Adam moments) across DP ranks — no replication. Each rank owns 1/DP of optimizer state, gathers as needed for the update.
- 3D rank mapping (TP, PP, DP) → 8 GPUs, respecting PCIe-switch topology (TP pairs closest, then PP, then DP).
- Full ~1B model now fits because params/optimizer are sharded across the mesh — compute the per-rank memory budget and confirm it fits 48GB.

## Oracle (write first)
- Full-mesh loss curve vs a reference (the M6 4-GPU run scaled, or a PyTorch/Megatron run) at BF16 tolerance.
- ZeRO-1 update math vs unsharded AdamW: identical results.

## Definition of done
1. Oracles pass — 8-GPU mesh trains the full 1B, loss matches reference.
2. **Headline numbers:** tokens/sec, MFU, memory high-water per rank, full comms breakdown across all three axes.
3. Checkpoint/resume works across the sharded mesh (resume the full 8-GPU run bit-exact).
4. `PROGRESS.md` updated; merged.

## Notes for Claude Code
- Three communicator groups now (TP, PP, DP). The DP all-reduce overlaps with backward compute (standard); make sure it doesn't contend with TP all-reduce on the same PCIe links — measure and document.
- This milestone will be allocation-gated: 8 L40S in one node may queue. Design the run to checkpoint frequently so a 16h slice makes real progress, and script auto-resubmit.
- Per-rank memory budget is the gating calculation — do it on paper before the run: (sharded weights + sharded grads + (1/DP) optimizer state + activations under 1F1B) < 48GB.
