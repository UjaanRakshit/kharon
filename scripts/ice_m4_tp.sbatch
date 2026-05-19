#!/bin/bash
#SBATCH -J kharon-m4tp
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH -N 1 -n 2
#SBATCH --mem=24G
#SBATCH -t 00:20:00
#SBATCH -o kharon-m4tp-%j.out

set -e
module load cuda/12.6.1 gcc/12.3.0 nvhpc-nccl/24.5 openmpi/4.1.8-cuda
export PATH=$CUDA_HOME/bin:$PATH
cd ~/scratch/kharon
[ -f data/input.bin ] || python scripts/prep_data.py data/input.bin
echo "=== build ==="; make all tp 2>&1 | tail -2

CFG="--layers 8 --d 512 --heads 8 --seq 256 --batch 32 --steps 300 --log 50 --lr 3e-4 --data data/input.bin"
echo "=== single-GPU reference (tp=1) ==="
./build/train.exe $CFG --ckptevery 0 --ckpt checkpoints/sg.ckpt | grep step
echo "=== TP=2 (loss curve should match; + comms cost) ==="
mpirun -n 2 ./build/tp_train.exe $CFG
