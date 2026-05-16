#!/bin/bash
#SBATCH -J kharon-m3
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1 -n 8
#SBATCH --mem=24G
#SBATCH -t 00:30:00
#SBATCH -o kharon-m3-%j.out

set -e
module load cuda/12.6.1 gcc/12.3.0
export PATH=$CUDA_HOME/bin:$PATH
cd ~/scratch/kharon

echo "=== node ==="; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "=== build ==="; make clean all >/dev/null 2>&1 && echo ok
[ -f data/input.bin ] || python scripts/prep_data.py data/input.bin
echo "=== oracles (L40S) ==="
for t in test_forward test_backward test_step test_resume test_flash test_bf16fwd test_bf16step; do
  printf "%-14s " "$t"; ./build/$t.exe >/dev/null 2>&1 && echo PASS || echo FAIL
done
echo "=== bf16 training (proxy d=512 L=8, 1000 steps) ==="
./build/train.exe --layers 8 --d 512 --heads 8 --seq 256 --batch 32 --steps 1000 \
  --log 100 --ckptevery 0 --lr 3e-4 --data data/input.bin --ckpt checkpoints/m3.ckpt
echo "=== bit-exact bf16 resume check ==="
mkdir -p checkpoints
./build/train.exe --layers 4 --d 256 --heads 4 --seq 128 --batch 16 --steps 100 \
  --log 100 --ckptevery 0 --data data/input.bin --ckpt checkpoints/r.ckpt | grep step
echo "-- resume 50 -> 100 should match loss above --"
./build/train.exe --layers 4 --d 256 --heads 4 --seq 128 --batch 16 --steps 50 \
  --log 100 --ckptevery 0 --data data/input.bin --ckpt checkpoints/r2.ckpt >/dev/null
./build/train.exe --layers 4 --d 256 --heads 4 --seq 128 --batch 16 --steps 100 \
  --log 100 --ckptevery 0 --resume --data data/input.bin --ckpt checkpoints/r2.ckpt | grep step
