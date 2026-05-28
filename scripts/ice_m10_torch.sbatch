#!/bin/bash
#SBATCH -J kharon-m10t
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1 -n 1
#SBATCH --mem=24G
#SBATCH -t 00:15:00
#SBATCH -o kharon-m10t-%j.out

set -e
module load pytorch/2.1.0
cd ~/scratch/kharon
echo "=== PyTorch baseline, same proxy config as Kharon single-GPU bf16 (d512 x 8L) ==="
python bench/torch_baseline.py --layers 8 --d 512 --heads 8 --vocab 256 --seq 256 --batch 32 --steps 100 --compile 1
echo "=== (Kharon C bf16, same config/L40S: 115k tok/s, M3 job 5348179) ==="
