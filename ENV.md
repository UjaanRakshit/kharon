# ENV.md — pinned toolchain

## Dev box (4060) — confirmed 2026-05-29
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Compute capability: **8.9 (Ada)** → `-arch=sm_89`
- VRAM: 8 GB (8188 MiB) — dev configs must fit this, keep batches/seq small
- Driver: 596.49
- CUDA toolkit: 12.6 (nvcc V12.6.20)
- Host compiler: MSVC v143 14.41.34120, from the standalone **Build Tools** install
  (Community has no C++ workload). vcvars:
  `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat`
  `cl.exe` is not on the default PATH — source vcvars64.bat first, then run nvcc/make.
- Toolchain verified 2026-05-29: `nvcc -arch=sm_89` compiled + ran a CUDA kernel on the 4060.

Note: the 4060 is Ada `sm_89`, the **same arch as the L40S**, not Turing `sm_75`.
One `-arch=sm_89` build serves both targets. (CONVENTIONS.md still says sm_75 for the
4060 — that line is wrong; flagged, not yet corrected.)

## Cluster (PACE ICE) — confirmed 2026-05-29
- Login: `login-ice.pace.gatech.edu` (RHEL 9.6). Access via SSH key only (no Duo for
  key auth); SSH multiplexing is blocked by the server. Account: `coc`.
- Storage: work under `~/scratch` (`/home/hice1/urakshit3/scratch`); repo at `~/scratch/kharon`.
- GPU partition: `ice-gpu`, `--gres=gpu:l40s:N` — 8× L40S (48 GB, Ada `sm_89`), sockets S:0-3.
  (also a100/a40/h100/h200/v100/rtx6000 on the same partition via gres type.)
- Toolchain modules: `cuda/12.6.1` (matches dev), `gcc/12.3.0`. The cuda module sets
  `$CUDA_HOME` but does NOT add bin to PATH — prepend it:
  `module load cuda/12.6.1 gcc/12.3.0 && export PATH=$CUDA_HOME/bin:$PATH`
- Build on RHEL9 is plain `make` (gcc host, no `-ccbin`/CCBIN). `make ARCH=sm_89`.
- NCCL: module `nvhpc-nccl/24.5` (also 25.5). NCCL_ROOT=.../nvhpc/24.5/.../comm_libs/nccl;
  sets CPATH/LD_LIBRARY_PATH. Build with `-lnccl`. MPI: `openmpi/4.1.8-cuda` for bootstrap.
- `nvidia-smi topo -m` (2× L40S, ice-gpu, node atl1-1-03-004-21): GPU0<->GPU1 = **PXB**
  (multiple PCIe bridges, does NOT cross the host bridge/SYS), both on NUMA 0, CPU aff 0-1.
  So a 2-GPU L40S alloc is a same-NUMA PXB PCIe pair (no NVLink) -> good for the TP=2 pair.
