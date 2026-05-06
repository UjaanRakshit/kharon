#!/bin/bash
#SBATCH -J kharon-m1
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1 -n 8
#SBATCH --mem=16G
#SBATCH -t 00:20:00
#SBATCH -o kharon-m1-%j.out

set -e
module load cuda/12.6.1 gcc/12.3.0
export PATH=$CUDA_HOME/bin:$PATH
cd ~/scratch/kharon

echo "=== node ==="; hostname; nvidia-smi -L
echo "=== topo ==="; nvidia-smi topo -m
echo "=== build ==="; make clean all >/dev/null 2>&1 && echo ok
echo "=== oracles (L40S) ==="
for t in test_forward test_backward test_step test_resume; do
  printf "%-14s " "$t"; ./build/$t.exe >/dev/null 2>&1 && echo PASS || echo FAIL
done
echo "=== benchmark (L40S) ==="
./build/bench_step.exe
