# ENV.md — pinned toolchain

## Dev box (4060) — confirmed 2026-05-29
- GPU: NVIDIA GeForce RTX 4060 Laptop GPU
- Compute capability: **8.9 (Ada)** → `-arch=sm_89`
- VRAM: 8 GB (8188 MiB) — dev configs must fit this, keep batches/seq small
- Driver: 596.49
- CUDA toolkit: 12.6 (nvcc V12.6.20)
- Host compiler: MSVC 2022 Community (`C:\Program Files\Microsoft Visual Studio\2022\Community`).
  `cl.exe` is not on the default PATH — build from the "x64 Native Tools Command Prompt
  for VS 2022", or point nvcc at it with `-ccbin "<...>\VC\Tools\MSVC\<ver>\bin\Hostx64\x64"`.

Note: the 4060 is Ada `sm_89`, the **same arch as the L40S**, not Turing `sm_75`.
One `-arch=sm_89` build serves both targets. (CONVENTIONS.md still says sm_75 for the
4060 — that line is wrong; flagged, not yet corrected.)

## Cluster (ICE, ice-gpu) — fill on first job
- GPU: 8× L40S (48 GB, Ada `sm_89`, PCIe Gen4, no NVLink)
- CUDA: __  NCCL: __  Driver: __
- Host compiler (gcc on RHEL9): __
- `module load` lines: __
- `nvidia-smi topo -m`: __ (which GPU pairs share a PCIe switch?)
