#!/bin/bash
#SBATCH -J kharon-m2
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1 -n 8
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o kharon-m2-%j.out

set -e
module load cuda/12.6.1 gcc/12.3.0
export PATH=$CUDA_HOME/bin:$PATH
cd ~/scratch/kharon

echo "=== node ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== build ==="; make clean all >/dev/null 2>&1 && echo ok
echo "=== oracles (L40S) ==="
for t in test_forward test_backward test_step test_resume test_flash; do
  printf "%-14s " "$t"; ./build/$t.exe >/dev/null 2>&1 && echo PASS || echo FAIL
done
echo "=== fused vs unfused ==="; ./build/bench_fused.exe
echo "=== flashattention (T=512) ==="; ./build/bench_flash.exe 4 8 512 64
echo "=== flashattention (T=1024) ==="; ./build/bench_flash.exe 2 8 1024 64
echo "=== full step ==="; ./build/bench_step.exe
