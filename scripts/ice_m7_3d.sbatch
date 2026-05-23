#!/bin/bash
#SBATCH -J kharon-m7
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:l40s:8
#SBATCH -N 1 -n 8
#SBATCH --mem=96G
#SBATCH -t 00:45:00
#SBATCH -o kharon-m7-%j.out

set -e
module load cuda/12.6.1 gcc/12.3.0 nvhpc-nccl/24.5 openmpi/4.1.8-cuda
export PATH=$CUDA_HOME/bin:$PATH
cd ~/scratch/kharon
[ -f data/input.bin ] || python scripts/prep_data.py data/input.bin
echo "=== build ==="; make all tp 2>&1 | tail -2

SMALL="--layers 8 --d 512 --heads 8 --seq 256 --mbatch 16 --data data/input.bin"
echo "=== oracle: single-GPU baseline over the global batch (M=32) ==="
mpirun -n 1 ./build/pp_train.exe $SMALL --tp 1 --dp 1 --M 32 --steps 200 --log 50
echo "=== DP grad-averaging cross-check: pure DP2 (2 GPU, M=16/replica) should match baseline ==="
mpirun -n 2 ./build/pp_train.exe $SMALL --tp 1 --dp 2 --M 16 --steps 200 --log 50
echo "=== full mesh TP2 x PP2 x DP2 (8 GPU) — loss tracks baseline; headline numbers ==="
mpirun -n 8 ./build/pp_train.exe $SMALL --tp 2 --dp 2 --M 16 --steps 200 --log 50

echo "=== checkpoint/resume bit-exact across the sharded 8-GPU mesh ==="
rm -f /tmp/m7ck.r*
echo "-- continuous 0->40 --"
mpirun -n 8 ./build/pp_train.exe $SMALL --tp 2 --dp 2 --M 16 --steps 40 --log 40
echo "-- split: 0->20 (save) then resume 20->40 --"
mpirun -n 8 ./build/pp_train.exe $SMALL --tp 2 --dp 2 --M 16 --steps 20 --log 20 --ckpt /tmp/m7ck
mpirun -n 8 ./build/pp_train.exe $SMALL --tp 2 --dp 2 --M 16 --steps 40 --log 40 --ckpt /tmp/m7ck

echo "=== full ~1B model on the 8-GPU mesh (memory + MFU headline) ==="
BIG="--layers 24 --d 2048 --heads 16 --seq 512 --mbatch 4 --M 8 --data data/input.bin"
mpirun -n 8 ./build/pp_train.exe $BIG --tp 2 --dp 2 --steps 30 --log 10
