#!/bin/bash
#SBATCH -J kharon-m4c
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:l40s:2
#SBATCH -N 1 -n 2
#SBATCH --mem=16G
#SBATCH -t 00:10:00
#SBATCH -o kharon-m4comms-%j.out

set -e
module load cuda/12.6.1 gcc/12.3.0 nvhpc-nccl/24.5 openmpi/4.1.8-cuda
export PATH=$CUDA_HOME/bin:$PATH
cd ~/scratch/kharon

echo "=== build tp ==="; make tp 2>&1 | tail -3
echo "=== nccl all-reduce on 2x L40S (PCIe PXB) ==="
mpirun -n 2 ./build/bench_allreduce.exe
