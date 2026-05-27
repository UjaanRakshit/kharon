#!/bin/bash
#SBATCH -J kharon-m9
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1 -n 1
#SBATCH --mem=32G
#SBATCH -t 00:30:00
#SBATCH -o kharon-m9-%j.out

set -e
module load cuda/12.6.1 gcc/12.3.0
export PATH=$CUDA_HOME/bin:$PATH
cd ~/scratch/kharon
echo "=== build ==="; make tests train 2>&1 | tail -2

echo "=== oracle: GRPO advantages + policy-gradient backward (== CE backward) ==="
./build/test_grpo.exe tests/m1_ref.bin

echo "=== SFT warm-start -> GRPO: single-digit addition, group sampling + prefix-shared rollouts ==="
./build/grpo_train.exe --layers 6 --d 512 --heads 8 --G 16 --prompts 32 --sft 200 --steps 400 --log 50 --lr 1e-3 --rllr 1e-4 --temp 0.7 --beta 0.02
