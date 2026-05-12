#!/bin/bash
# ============================================================
# CoDMD Training Script
# ============================================================
# Prerequisites:
#   - Set PYTHONPATH to the repo root
#   - Activate your conda environment
#   - Adjust --nnodes, --nproc_per_node, tar_data_dir in config
# ============================================================

export PYTHONPATH=$(dirname $(dirname $(realpath $0))):$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# --- Wan2.1-T2V-1.3B (4-step, 32 GPUs) ---
torchrun --nnodes 4 --nproc_per_node=8 --rdzv_id=5235 \
    copula_dmd/train_dmd.py -- \
    --config_path configs/wan_dmd_tar.yaml

# --- Wan2.1-T2V-14B (4-step, 32 GPUs) ---
# torchrun --nnodes 4 --nproc_per_node=8 --rdzv_id=5235 \
#     copula_dmd/train_dmd.py -- \
#     --config_path configs/wan_dmd_tar_14b.yaml
