#!/bin/bash
# gpu-bind.sh: 1 rank per GPU, pin CPU+MEM near that GPU
set -euo pipefail

# added by me
export MPICH_MALLOC_FALLBACK=1
ulimit -s unlimited

LOCAL_RANK=${SLURM_LOCALID:-0}

# If Slurm assigned GPUs, use that; else map LOCAL_RANK -> GPU index.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  gpu=0  # Slurm already isolated one GPU per task
else
  gpu=${LOCAL_RANK}                    # simple 0..3 mapping
  export CUDA_VISIBLE_DEVICES=${gpu}
fi

# Assume NUMA node index matches GPU index (common on 4× GH nodes).
# If that’s not true on your system, switch to --gpu-bind=closest, or map explicitly.
numa=${LOCAL_RANK}

exec numactl --cpunodebind=${numa} --membind=${numa} "$@"

