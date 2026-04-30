# Milestone 4 — Tensor parallelism (TP=2) on L40S

**Window:** Months 4–5 · **Hardware:** 2× L40S (one node) · **Branch:** `m4-tp`
**Prereq:** M3 · **Read first:** `CONVENTIONS.md`

## Goal
Split each transformer layer across 2 GPUs (Megatron-style tensor parallelism) with hand-orchestrated NCCL collectives, over L40S PCIe Gen4 (no NVLink). First real multi-GPU correctness + first interconnect measurement.

## Scope
- **Column-parallel** QKV + MLP-in; **row-parallel** attention-out + MLP-out (the Megatron sharding).
- Collectives: all-reduce after row-parallel, all-gather where needed — via NCCL, hand-placed.
- Rank↔GPU↔socket pinning: use `nvidia-smi topo -m` to put the TP pair on GPUs sharing the nearest PCIe switch; pin each rank to that GPU's NUMA socket (`S:0-3`).
- Setup NCCL init / process-per-GPU launch (MPI or your own bootstrap).

## Oracle (write first)
- TP=2 forward+backward vs the single-GPU M3 path: same logits/grads at BF16 tolerance, same loss curve.

## Definition of done
1. Oracle passes — TP=2 loss curve matches single-GPU.
2. **TP comms measured on L40S PCIe:** time in collectives vs compute, and comms/compute overlap %. This is the headline number — PCIe TP is bandwidth-constrained and the data showing *how much* is a key result.
3. Tokens/sec vs single-GPU (expect sublinear due to PCIe collectives — that's the finding, not a bug).
4. `PROGRESS.md` updated; merged.

## Stretch
- Repeat the TP=2 run on the **A100×2 (NVLink) node** and record the delta. L40S-PCIe vs A100-NVLink on identical TP=2 is a clean, publishable interconnect-impact result.

## Notes for Claude Code
- Overlap is the whole game on PCIe: launch the next compute while the all-reduce is in flight. Measure with and without overlap to quantify what overlap buys.
- Don't assume NVLink anywhere in the L40S path — every collective is PCIe. Code/comments should reflect that.
