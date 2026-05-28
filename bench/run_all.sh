#!/bin/bash
# One entrypoint to regenerate the Kharon performance map. Local steps run anywhere with a
# CUDA GPU + toolchain; cluster steps are the sbatch jobs whose numbers feed results.json.
# Plots are regenerated purely from results.json, so `python bench/plot.py` alone refreshes
# every figure from the recorded numbers.
set -e
cd "$(dirname "$0")/.."

echo "== build =="
make all >/dev/null

echo "== local microbenchmarks (dev GPU) =="
./build/bench_bf16.exe        || true   # bf16 tensor-core GEMM vs fp32 cuBLAS (3.3x)
./build/bench_flash.exe       || true   # hand FlashAttention vs cuBLAS-naive
./build/bench_infer.exe --layers 6 --d 768 --heads 12 --vocab 50257 --seq 1024 --prompt 128 --new 256 || true

echo "== regenerate plots from results.json =="
python bench/plot.py

cat <<'EOF'

== cluster series (run via scripts/, numbers recorded in bench/results.json) ==
  scripts/ice_m6_tppp.sbatch    TP/PP/TP×PP scaling + comms breakdown (4 GPU)
  scripts/ice_m7_3d.sbatch      full 8-GPU TP×PP×DP mesh, ZeRO-1, 1.2B (MFU, mem)
  scripts/ice_m8_infer.sbatch   paged-KV inference throughput + KV-mem efficiency
  scripts/ice_m9_grpo.sbatch    GRPO SFT->RL reward curve
  scripts/ice_m10_torch.sbatch  PyTorch baseline (same config, L40S)

See bench/REPORT.md for the full performance map.
EOF
