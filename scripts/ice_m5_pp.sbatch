#!/bin/bash
#SBATCH -J kharon-m5pp
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:l40s:4
#SBATCH -N 1 -n 4
#SBATCH --mem=48G
#SBATCH -t 00:25:00
#SBATCH -o kharon-m5pp-%j.out

set -e
module load cuda/12.6.1 gcc/12.3.0 nvhpc-nccl/24.5 openmpi/4.1.8-cuda
export PATH=$CUDA_HOME/bin:$PATH
cd ~/scratch/kharon
[ -f data/input.bin ] || python scripts/prep_data.py data/input.bin
echo "=== build ==="; make all tp 2>&1 | tail -2

BASE="--layers 8 --d 512 --heads 8 --seq 256 --mbatch 16 --data data/input.bin"
echo "=== oracle: P=1 vs P=2 loss curves (should match, bf16 tol) ==="
mpirun -n 1 ./build/pp_train.exe $BASE --M 16 --steps 200 --log 50
mpirun -n 2 ./build/pp_train.exe $BASE --M 16 --steps 200 --log 50
echo "=== bubble vs M at P=2 ==="
for MB in 8 16 32 64; do mpirun -n 2 ./build/pp_train.exe $BASE --M $MB --steps 1 --log 1000; done
echo "=== bubble vs M at P=4 ==="
for MB in 16 32 64; do mpirun -n 4 ./build/pp_train.exe $BASE --M $MB --steps 1 --log 1000; done
