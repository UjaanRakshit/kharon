#!/bin/bash
#SBATCH -J kharon-m6
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:l40s:4
#SBATCH -N 1 -n 4
#SBATCH --mem=48G
#SBATCH -t 00:25:00
#SBATCH -o kharon-m6-%j.out

set -e
module load cuda/12.6.1 gcc/12.3.0 nvhpc-nccl/24.5 openmpi/4.1.8-cuda
export PATH=$CUDA_HOME/bin:$PATH
cd ~/scratch/kharon
[ -f data/input.bin ] || python scripts/prep_data.py data/input.bin
echo "=== build ==="; make all tp 2>&1 | tail -2

BASE="--layers 8 --d 512 --heads 8 --seq 256 --mbatch 16 --M 16 --data data/input.bin"
echo "=== oracle: single-GPU (PP1,TP1) baseline ==="
mpirun -n 1 ./build/pp_train.exe $BASE --tp 1 --steps 200 --log 50
echo "=== TP2 x PP2 (4 GPU) — loss should match baseline; + comms breakdown ==="
mpirun -n 4 ./build/pp_train.exe $BASE --tp 2 --steps 200 --log 50
echo "=== cross-check: pure TP2 (2 GPU, PP1) and pure PP2 (2 GPU, TP1) ==="
mpirun -n 2 ./build/pp_train.exe $BASE --tp 2 --steps 1 --log 1000
mpirun -n 2 ./build/pp_train.exe $BASE --tp 1 --steps 1 --log 1000
