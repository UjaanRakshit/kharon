#!/bin/bash
#SBATCH -J kharon-m8
#SBATCH -A coc
#SBATCH -p ice-gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -N 1 -n 1
#SBATCH --mem=32G
#SBATCH -t 00:20:00
#SBATCH -o kharon-m8-%j.out

set -e
module load cuda/12.6.1 gcc/12.3.0
export PATH=$CUDA_HOME/bin:$PATH
cd ~/scratch/kharon
echo "=== build ==="; make tests bench 2>&1 | tail -2

echo "=== oracle: paged-KV continuous batching == PyTorch greedy (fp32, token-exact) ==="
# gen_ref.bin is generated locally (torch not on compute nodes) and shipped with the tree.
./build/test_infer.exe tests/gen_ref.bin

echo "=== rollout throughput + KV mem efficiency (L40S, ~350M proxy, GPT-2-large-ish) ==="
./build/bench_infer.exe --layers 24 --d 1024 --heads 16 --vocab 50257 --seq 1024 --prompt 128 --new 256 --block 16 --bf16 1
echo "=== larger model (d2048 x 24L ~1.2B), batch rollouts ==="
./build/bench_infer.exe --layers 24 --d 2048 --heads 16 --vocab 50257 --seq 1024 --prompt 256 --new 256 --block 16 --bf16 1
echo "=== block-size sweep (indirection vs waste) at G=16 ==="
for B in 8 16 32 64; do
  ./build/bench_infer.exe --layers 24 --d 1024 --heads 16 --vocab 50257 --seq 1024 --prompt 128 --new 256 --block $B --bf16 1 | grep "G=16:"
done
