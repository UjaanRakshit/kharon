# Milestone 6 — Compose TP × PP (4 GPU)

**Window:** Months 7–8 · **Hardware:** 4× L40S (one node) · **Branch:** `m6-tp-pp`
**Prereq:** M4, M5 · **Read first:** `CONVENTIONS.md`

## Goal
Run tensor parallelism and pipeline parallelism together: TP=2 within each pipeline stage, PP=2 across stages = 4 GPUs. This is where the two collective patterns must coexist without stomping on each other — the real systems-composition challenge.

## Scope
- 2D rank layout: a (TP, PP) grid. Define rank→GPU mapping with TP pairs on the closest PCIe-switch GPUs, PP crossing the switch.
- Two NCCL communicator groups: TP-group (all-reduce within stage) and PP-group (point-to-point across stages). They must not deadlock or serialize each other.
- Combine M4's TP collectives inside each M5 pipeline stage.

## Oracle (write first)
- TP×PP loss curve vs single-GPU M3 path at BF16 tolerance.

## Definition of done
1. Oracle passes — 4-GPU loss matches single-GPU.
2. Tokens/sec and MFU on 4× L40S recorded.
3. Combined comms breakdown: TP all-reduce time vs PP send/recv time vs compute; show they overlap rather than serialize.
4. Memory high-water per rank.
5. `PROGRESS.md` updated; merged.

## Notes for Claude Code
- The deadlock risk is real: ordering of TP all-reduce vs PP recv across ranks must be consistent. Use separate CUDA streams per communicator and document the ordering invariant.
- Build the rank-mapping logic to be parameterized by (TP, PP, DP) now — M7 adds DP and you don't want to rewrite it.
