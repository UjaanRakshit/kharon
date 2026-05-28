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
| 6 | TP×PP | done | ✓ TP2xPP2 loss matches single-GPU | 4-GPU 268k tok/s (2.35x); TP all-reduce dominates | overlap is future (1 stream) |
| 7 | +ZeRO-1 DP (8 GPU) | done | ✓ ZeRO update bit-identical; mesh loss tracks baseline; resume bit-exact | 1.2B on 8 GPU: 30.5k tok/s, MFU 15.3%, 19.1GB/rank | DP all-gather is fp32 (future: bf16 gather) |
| 8 | Inference engine | done | ✓ paged decode == PyTorch greedy (token-exact); paged == contiguous | 1.2B G=32 1606 tok/s; paged KV 2x; prefix saves 1.5GB @G=32 | paged-attn is 1 block/(tok,head) (future: tiled) |
| 9 | GRPO loop | done | ✓ advantages vs formula; GRPO bwd == CE bwd (bit-exact) + linearity | reward curve moves (0→0.65 reward, acc 0→~12%+); prefix-shared rollouts | exact-match is hard for the proxy (shaped reward) |
| 10 | Benchmark + writeup | done | — (consolidation) | bench/REPORT.md + 5 plots; PyTorch eager 1.53x faster single-GPU (honest loss) | official-FA/vLLM/Megatron head-to-heads install-gated |

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
- Kharon C vs PyTorch eager, single-GPU, SAME proxy (d512x8L seq256 batch32 bf16, L40S job
  5348832): PyTorch eager 176k tok/s (46.5 ms) vs Kharon 115k tok/s -> PyTorch 1.53x FASTER
  (Kharon 0.65x). The honest end-to-end loss: PyTorch uses fused tensor-core bf16 SDPA;
  Kharon's attention is hand-written FP32 (0.82x cuBLAS-naive). bf16 GEMMs match vendor.
  torch.compile failed (inductor backend error in the pytorch/2.1.0 module env). Top fix:
  tensor-core bf16 FlashAttention (the FA kernel doesn't yet use the M3 GEMM path).
- FlashAttention vs official: official FA is fused bf16/tensor-core (inside PyTorch SDPA above);
  ours is FP32 warp-per-row 0.82x cuBLAS-naive. Direct standalone head-to-head install-gated.
- Custom GEMM vs cuBLAS: we call cuBLAS (cublasGemmEx) for the GEMMs; bf16 vs fp32 = 3.3x.
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
- TP x PP (4x L40S, proxy d512 x 8L): TP2xPP2 loss matches single-GPU (bf16). Scaling
  (tok/s): 1GPU 114k -> TP2 146k (1.28x) -> PP2 217k (1.90x) -> TP2xPP2 268k (2.35x).
  Step breakdown (TP2xPP2): compute 172ms + TP all-reduce 52ms (21%) + PP bubble/comms 20ms
  (8%). Finding: on PCIe (no NVLink) TP all-reduce dominates; PP scales far better. Two NCCL
  comms (TP sub-comm via ncclCommSplit + PP p2p on global comm) coexist w/o deadlock.
- Full 8-GPU mesh (TP2 x PP2 x DP2, ZeRO-1), full ~1.2B model (d2048 x 24L, seq512):
  1072 ms/step, 30.5k tok/s, MFU 15.3% (L40S bf16 dense peak 181 TFLOP/s). 19.1 GB/rank
  used (of 47.7) -> full 1B fits with room. Step breakdown: compute 729 + TP all-reduce 130
  (12%) + PP bubble/comms 111 (10%) + DP opt-comm 102 ms (9.5%). All three axes cost ~10%
  each; the honest map is "PCIe-comms-bound, ~32% in collectives, no axis free". Bubble
  11.5% vs theory 11.1% (M=8,P=2). On the tiny proxy (d512 x 8L) the mesh is far more
  comms-bound: 531k tok/s but MFU only 5.6%, TP all-reduce alone 21% (compute too small to
  hide PCIe) — small models do not amortize the interconnect.
- ZeRO-1 (shard Adam m,v across DP): opt-state/rank 1215 MB at DP=2 vs 2430 MB replicated
  (exactly 1/DP). fp32 master kept replicated + reconstructed by all-gather each step (the
  DP opt-comm cost above); sharding the master too is the remaining ZeRO-1 increment.
  Update math bit-identical to unsharded AdamW (test_zero, incl. uneven padding).
- DP grad-averaging equivalence (2x L40S): pure DP2 over M=16/replica matches single-GPU
  over M=32 to bf16 tol (loss 2.371 vs 2.359 at step 200) — reduce-scatter(avg) is exact.
- Checkpoint/resume bit-exact on the sharded 8-GPU mesh: continuous 0->40 == split (save@20,
  resume->40), loss 3.2218 identical to 4 d.p. (per-rank ckpt stores the moment shard).
- M8 inference engine (paged-KV + continuous batching). Oracle (L40S + 4060): fp32 paged
  decode == PyTorch greedy token-for-token across staggered batch entry/exit; paged ==
  contiguous (block_size>=len); bf16 greedy matches fp32 here too. Rollout throughput
  (L40S, bf16, prompt128+new256): 350M (d1024 x 24L) G=1/8/16/32 = 166/810/1619/2693 tok/s;
  1.2B (d2048 x 24L) = 82/531/941/1606 tok/s. Continuous batching scales ~16x from G=1->32
  (decode is memory-bw/latency bound; batching amortizes weight reads). Paged KV vs naive
  contiguous (reserve full seq): 2.0-2.67x less memory (only used blocks allocated). Prefix
  sharing (one prompt, G samples = the GRPO pattern): blocks stored once, saving (G-1)*full
  prefix blocks — 1.56 GB at G=32 on the 1.2B model (256-tok prompt). Only FULL prefix
  blocks shared read-only; partial last block is copy-on-write per sequence (vLLM-style).
  Engine reuses the bf16 forward kernels; q/k/v read straight from the fused qkv buffer
  (no head transpose). paged-attn is one block per (token,head) — correct but not tiled;
  a blocked/warp-efficient kernel is the obvious throughput lever (noted, not done).
- M9 GRPO (no critic) on byte-level single-digit addition. Oracles: group-relative
  advantages == direct (r-mean)/std; the policy-gradient backward is bit-identical to the
  PyTorch-validated CE backward at coef=1/N (since dlogits = coef*(probs-onehot)) and
  scales linearly in coef. Key finding: GRPO from a *random* init collapses to the
  marginal-mode answer — once collapsed, every group has zero reward-variance, so the
  advantage is 0 and there is no escape (verified: greedy stuck at the modal token). Real
  GRPO sharpens a pretrained policy, so the loop is SFT warm-start (CE, 0->96%) -> snapshot
  as frozen KL reference -> GRPO at lr/10. GRPO then refines SFT's ~87-96% -> 100% task
  accuracy (mean reward 1.05->1.10, stable), sample completions all correct (no reward
  hacking). RL lr must be << SFT lr: at the SFT lr the policy-gradient noise destroys the
  SFT solution (96%->29%). Rollouts via the M8 engine, prefix-shared per group (30
  blocks/step saved at G=16, bs=2), ~51k tok/s on L40S. Handoff: model_sync_bf16 after each
  update so the engine serves the current policy. Shaped/closeness rewards were removed —
  they create a hackable constant-output optimum the sample logging caught. (Cluster job
  5348827: oracle PASS, SFT 0->87%, GRPO ->100%.)
- vs Megatron-LM same config: __
- GRPO reward curve delta: __

## Session log
<!-- newest first: date — what was done — what's next -->
- 2026-05-30 (autonomous) — M10 DONE. The honest performance map: bench/REPORT.md (wins/losses
  + why), bench/results.json (single source of truth for every number), bench/plot.py -> 5 plots
  checked in (scaling, comms breakdown, bubble, inference, GRPO), bench/run_all.sh (one
  entrypoint), README.md headline + reproduction. Ran a real PyTorch baseline on L40S (job
  5348832, same proxy): PyTorch eager 176k vs Kharon 115k tok/s -> PyTorch 1.53x faster
  end-to-end, the honest single-GPU loss (fused tensor-core SDPA vs our FP32 FA). Map summary:
  WIN bf16 GEMM (3.3x) + systems work (paged-KV 2-2.67x mem, 8-GPU ZeRO-3D mesh MFU 15.3%, GRPO
  ->100%); LOSE the attention kernel single-GPU; interconnect-bound (PCIe no-NVLink) at scale.
  Open (honest): official-FA/vLLM/Megatron head-to-heads + A100-NVLink TP study are install/
  allocation-gated. m10-bench merged. ALL 10 MILESTONES DONE.
- 2026-05-30 (autonomous) — M9 DONE, validated on L40S (job 5348827) + 4060. GRPO (no critic)
  on byte-level single-digit addition. Reused the M7 trainer (policy-gradient via a new
  model_grpo_backward: dlogits = coef*(probs-onehot), coef = mask*(advantage - beta*KL)/N;
  factored the bf16 backward into a shared tail) and the M8 engine for prefix-shared group
  rollouts (added temperature sampling). Oracles: advantages vs formula; GRPO bwd bit-
  identical to the PyTorch-validated CE bwd + linearity (test_grpo). Hit + diagnosed the
  cold-start collapse (random policy -> marginal mode -> zero-variance groups -> dead
  gradient) and reward hacking (closeness reward -> constant output, caught by sample
  logging). Fix = SFT warm-start (0->96%) then GRPO at lr/10 refines 96->100%, KL to the
  SFT snapshot. Also fixed a real bug: the global cublas handle wasn't refcounted, so a
  process with >1 model + engine double-freed it (heap corruption) — now refcounted.
  m9-grpo merged. NEXT: M10 (benchmark suite + writeup — the honest perf map).
- 2026-05-30 (autonomous) — M8 DONE, validated on L40S (job 5348786) + 4060. Real paged-KV
  inference engine: fixed-size block pool + per-seq block tables (vLLM-style), decode forward
  reusing the bf16 kernels (q/k/v read straight from fused qkv, no transpose), continuous-batch
  scheduler (staggered entry/exit, blocks freed on EOS), greedy sampling, prefix sharing for
  the shared-prompt (GRPO) case with copy-on-write of the partial block. New kernels
  (embed_pos, append_kv, paged_attn, gather_rows) templated fp32/bf16. Oracle: fp32 paged
  decode == PyTorch greedy token-exact (4 prompts, staggered), paged == contiguous, prefix
  group matches. Numbers: 1.2B G=32 1606 tok/s (16x scaling G1->32); paged KV 2-2.67x vs
  naive; prefix sharing saves 1.5 GB at G=32. test_infer green locally + cluster. m8-infer
  merged. Finding: paged-attn (1 block/(tok,head)) is the throughput lever for later; decode
  is batching-bound, prefix sharing is a big win for group rollouts. NEXT: M9 (GRPO loop).
- 2026-05-30 (autonomous) — M7 DONE, validated on 8x L40S (job 5348750, 10 min walltime).
  ZeRO-1 data parallel + full TP2 x PP2 x DP2 mesh. comms_init_grid3 adds a DP sub-comm
  (ncclCommSplit); rank = dp*(PP*TP)+pp*TP+tp keeps TP pairs adjacent (closest PCIe), then
  PP, then DP. ZeRO-1: reduce-scatter(avg) fp32 grads -> sharded AdamW on the master slice ->
  all-gather params; Adam moments sharded to 1/DP (1215 MB vs 2430 at DP=2). Trained the full
  ~1.2B model (d2048 x 24L) on 8 GPU: 30.5k tok/s, MFU 15.3%, 19.1 GB/rank (fits 48). 3-axis
  breakdown each ~10% (TP 130 / PP 111 / DP 102 ms over 729 compute) -> PCIe-comms-bound, no
  free axis. Oracles: ZeRO update bit-identical to unsharded (test_zero); DP2-over-M16 ==
  1GPU-over-M32 (bf16 tol); resume bit-exact across the sharded mesh (3.2218 == 3.2218).
  m7-3d merged. Finding: ZeRO all-gather of the fp32 master each step is a real PCIe cost;
  sharding the master + bf16 gather is the next increment. NEXT: M8 (inference engine:
  paged KV, continuous batching).
- 2026-05-30 (autonomous) — M6 DONE, validated on 4x L40S (job 5348719). Composed TP x PP:
  2D rank grid (rank = pp_stage*tp + tp_rank), TP sub-comm via ncclCommSplit + PP p2p on
  global comm (peer = rank +- tp). Pipeline stages are TP-sharded (forward_pp/backward_pp
  made tp-aware), double-sharded init (layer x tp). TP2xPP2 loss matches single-GPU (bf16);
  4-GPU 2.35x; breakdown shows TP all-reduce (52ms) dominates, PP bubble (20ms) cheap. tp=1
  and pp=1 degenerate cases bit-validated locally (test_tp1/test_pp1). m6-tp-pp merged.
  NEXT: M7 (+ZeRO-1 DP -> full TP*PP*DP 8-GPU mesh). Note: per-rank mem high-water available
  via arena instrumentation (not re-measured); comms on 1 stream (overlap = future opt).
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
