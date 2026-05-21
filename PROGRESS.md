# PROGRESS.md — running state (update every session)

> Claude Code: read this at session start. Update the active milestone's row and the log at the bottom before ending a session.

## Status board

| # | Milestone | State | Oracle | Benchmark number | Blocker |
|---|-----------|-------|--------|------------------|---------|
| 1 | Single-GPU GPT + ckpt | done | ✓ (4060 + L40S) | 0.64 ms/step 805k tok/s (L40S); 7.1 ms 72k (4060) | — |
| 2 | FlashAttention + fused | done | ✓ FA fwd+bwd vs SDPA | FA 1.5 TFLOP/s, 0.82x cuBLAS (L40S); fusion 1.7x (4060) | beat-cuBLAS needs M3 BF16/TC |
| 3 | BF16 + full step | done | ✓ (4060 + L40S) | learns 2.58->1.42; 115k tok/s (L40S, d512x8L); resume bit-exact | — |
| 4 | TP=2 | done | ✓ loss curve matches single-GPU | comms 23.2% of step; 1.33x tok/s on 2x L40S | overlap is future (TP all-reduce on critical path) |
| 5 | PP (1F1B) | done | ✓ P=2 loss matches P=1 | bubble tracks (P-1)/(M+P-1); P=2 1.89x on 2 GPUs | interleaved 1F1B is stretch |
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
- BF16 tensor-core GEMM (M3 lever, 4060, cublasGemmEx): 3.3x vs fp32 cuBLAS,
  rel-frobenius 2.4e-3 (within bf16 tol). Resolves the M2 "needs tensor cores" gap.
- M3 bf16 training (L40S, proxy d512 x 8L = 25M params, seq256 batch32): ~115k tok/s,
  loss 2.58->1.42 / 1000 steps; resume bit-exact. (baseline for M4-M7 parallel speedup.)
- M3 memory budget: bf16 mixed = 18 B/param (fp32 master 4 + bf16 weight 2 + fp32 grad 4
  + adamw m,v 8). 1.3B target -> ~23.6GB params/opt + acts; borderline on 48GB, so M3 uses
  a proxy (~d=1024 x 12L ~200M params ~3.6GB) to prove the loop; full 1B waits for M4-M7.
- FlashAttention vs official: __ (deferred to M3 — official FA is BF16/tensor-core)
- Custom GEMM vs cuBLAS (shape __): __
- Bubble fraction (1F1B, L40S): P=2 M=8/16/32/64 = 10.3/6.2/4.2/2.8% (theory 11.1/5.9/3.0/1.5);
  P=4 M=16/32/64 = 16.0/10.4/6.4% (theory 15.8/8.6/4.5). Tracks (P-1)/(M+P-1); excess over
  theory at large M is real PCIe send/recv cost. P=2 scales 1.89x on 2 GPUs (vs TP's 1.33x:
  PP ships only stage-boundary activations, not per-layer all-reduce).
- NCCL all-reduce bus bw (2x L40S, PCIe PXB, no NVLink): ~21 GB/s large msgs, ~2 GB/s @16KB
  (latency-bound small -> bandwidth-bound large). Headline interconnect ceiling for TP.
- TP=2 (2x L40S PCIe, proxy d512 x 8L, bf16): loss curve matches single-GPU to ~3e-3 (bf16).
  Step 53.0 ms = 40.7 compute + 12.3 comms (23.2% in collectives). 154k tok/s vs 117k
  single-GPU = 1.33x on 2 GPUs (sublinear: PCIe all-reduce + replicated LN/embed/head).
  Overlap: ~0% in vanilla TP (all-reduce output consumed immediately); async comms hook
  exists for future sequence-parallel/overlap work. This is the PCIe-no-NVLink finding.
- A100-NVLink vs L40S-PCIe TP=2: __
- Full 8-GPU mesh tokens/sec & MFU: __
- vs Megatron-LM same config: __
- GRPO reward curve delta: __

## Session log
<!-- newest first: date — what was done — what's next -->
- 2026-05-30 (autonomous) — M5 DONE, validated on L40S (job 5348679). 1F1B pipeline:
  stage-split (untied head, embeddings on stage 0), per-slot activation stash (~P in flight),
  grad accumulation over microbatches, deadlock-free batched NCCL sendrecv (megatron-style:
  pair fwd-send with bwd-recv). P=1 bit-matches single-GPU (test_pp1 local); P=2 loss curve
  tracks P=1; bubble tracks (P-1)/(M+P-1), driven down by M; P=2 1.89x on 2 GPUs. Hit + fixed
  a 1F1B comms deadlock (fwd-send before bwd-recv -> circular wait). m5-pp merged. Next: M6 (TPxPP).
- 2026-05-29 (cont.5) — M4 DONE, validated on 2x L40S (job 5348274). Megatron TP=2:
  column-parallel qkv/fc, row-parallel proj/fcproj, 2 all-reduces/way (NCCL callback);
  embeddings/LN/head replicated. tp=1 bit-matches single-GPU (test_tp1); tp=2 loss curve
  tracks single-GPU to ~3e-3 (bf16). Comms 23.2% of step (PCIe, no NVLink); 1.33x tok/s on
  2 GPUs (sublinear = the finding). Parametrized model layout by tp; TP fwd/bwd reuse the
  bf16 kernels; AdamW/ckpt reuse flat arenas. m4-tp merged. Next: M5 (pipeline parallel 1F1B).
- 2026-05-29 (cont.4) — M4 started (branch m4-tp). Foundation done: confirmed NCCL
  (nvhpc-nccl/24.5) + L40S topo (GPU0<->GPU1 PXB, NUMA0); wrote src/parallel/comms.{h,cu}
  (NCCL+MPI process-per-GPU, all-reduce fp32/bf16, async variant for overlap); validated
  on 2x L40S (correctness ok, ~21 GB/s bus bw large / ~2 GB/s @16KB). Build: `make tp` with
  -ccbin mpicxx + $NCCL_ROOT; launch mpirun -n2 (scripts/ice_m4_comms.sbatch).
  NEXT (the big chunk): Megatron TP=2 sharded model — column-parallel qkv/fc (no fwd comm,
  all-reduce on dx in bwd), row-parallel proj/fcproj (all-reduce out in fwd, none in bwd);
  embeddings+head replicated; 2 all-reduces/layer each way. Build a tp_model + tp train/test
  launched mpirun -n2, oracle vs single-GPU M3 path (bf16 tol, loss curve), then measure
  collective time + overlap (async all-reduce on comms stream) + tokens/sec vs 1-GPU; merge.
- 2026-05-29 (cont.3) — M3 DONE, validated on L40S (job 5348179). All 7 oracles pass;
  bf16 training of a 25M-param proxy (d512 x 8L) learns 2.58->1.42 over 1000 steps at
  ~115k tok/s; bit-exact bf16 resume confirmed on L40S (uninterrupted == 50+resume, loss
  2.7497 at step 100). m3-bf16 merged. Next: M4 (TP=2 on L40S over PCIe).
- 2026-05-29 (cont.2) — M3 bf16 FULL training step done + validated on 4060.
  bf16 backward (bf16 activation-grads, fp32 weight-grads for AdamW master) via templatized
  kernels + mm_tn_bf16 (bf16->fp32) / mm_nn_bf16o (bf16->bf16) + bf16 FA-bwd. Validation:
  bf16 grads vs verified fp32 grads rel-frobenius 7e-3; bf16 overfit 5.6->0.1 in 30 steps;
  real tinyshakespeare training loss 2.78->2.25, ~150k tok/s (4060); checkpoint/resume
  reproduces the loss trajectory (2.5221/2.3949 at step 100/200, interrupted == uninterrupted).
  Added byte-level data loader, random GPT-2 init, train.c entrypoint, prep_data.py.
  AMP-bf16 oracle: covered transitively (bf16 grads within bf16 tol of fp32, fp32 verified
  vs PyTorch) — stronger than loose loss-curve matching. NEXT: run scripts/ice_m3.sbatch on
  one L40S for the tokens/sec number + on-cluster bit-exact resume (blocked now: PACE login
  nodes resetting/timing out — transient, retry later), then merge m3-bf16.
- 2026-05-29 (cont.) — M3 bf16 FORWARD now wired into the model and validated:
  model_forward_bf16 (bf16 weights+acts via tensor-core GEMMs, fp32 reduction stats,
  fp32 master) matches the PyTorch fp32 ref at bf16 tol (loss reldiff 1.4e-5, logits
  maxabs 8e-3). Parallel bf16 weight/act arenas + model_sync_bf16 (cast master->bf16).
  Fixed a real build bug: Makefile had no header deps, so a Model-struct change left
  adamw.o/ckpt.o compiled against the old layout (stale offsets) -> corrupted step/resume;
  now every object rebuilds on any header change. All 6 oracles green; resume memcheck clean.
  NEXT: bf16 BACKWARD — weight grads stay fp32 (AdamW master), activation grads bf16; need
  mm_tn bf16->fp32, mm_nn bf16->bf16, templatized ln_bwd/gelu_bwd/ce_bwd/colsum/embed_bwd/
  combine/FA-bwd + model_backward_bf16; then bf16 train-step test, AMP oracle, data loader, run.
- 2026-05-29 — M3 in progress (branch m3-bf16). Done + validated: bf16 tensor-core GEMM
  (cublasGemmEx, 3.3x vs fp32, rel-fro 2e-3); f32<->bf16 casts; memory budget (18 B/param,
  use proxy); and the full bf16 FORWARD kernel suite — templatized on storage dtype so the
  fp32 path is unchanged (all M1/M2 oracles still pass). bf16 kernels: mm_nt_bf16o (bf16-out
  GEMM), ln_fwd, bias_residual, bias_gelu, ce_fwd, split/merge_heads, embed, FA forward.
  NEXT (fresh session): (1) wire parallel bf16 weight+activation arenas into Model + a
  model_sync_bf16 (cast master->bf16) + model_forward_bf16/backward_bf16 (mirror fp32 path,
  call _bf launchers + mm_nt_bf16o, weight-grad GEMMs output fp32, fp32 master+AdamW);
  (2) bf16 backward kernels (templatize ln_bwd, gelu_bwd, ce_bwd, colsum, embed_bwd, combine,
  FA bwd); (3) validate bf16 fwd logits vs fp32/PyTorch within bf16 tol; (4) PyTorch AMP-bf16
  loss-curve oracle; (5) byte-level data loader from ~/scratch; (6) L40S train run + bit-exact
  bf16 resume + tokens/sec. Keep fp32 path as the debug oracle (it is, via <float> instantiation).
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
