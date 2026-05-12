#!/bin/bash
# ============================================================
# CoDMD Inference Script
# ============================================================
# Prerequisites:
#   - Set PYTHONPATH to the repo root
#   - Activate your conda environment
#   - Prepare a prompts.txt file (one prompt per line)
# ============================================================

export PYTHONPATH=$(dirname $(dirname $(realpath $0))):$PYTHONPATH

CHECKPOINT_DIR="<PATH_TO_CHECKPOINT>/checkpoint_model_003000"
OUTPUT_DIR="./results"
PROMPT_FILE="prompts.txt"
CONFIG="configs/wan_dmd_tar.yaml"     # or configs/wan_dmd_tar_14b.yaml

# --- Single GPU inference ---
# python inference.py \
#     --config_path $CONFIG \
#     --checkpoint_folder $CHECKPOINT_DIR \
#     --output_folder $OUTPUT_DIR \
#     --prompt_file_path $PROMPT_FILE \
#     --num_seeds 5

# --- Multi-GPU DDP inference (8 GPUs) ---
torchrun --nproc_per_node=8 --master_port=29600 \
    inference.py \
    --config_path $CONFIG \
    --checkpoint_folder $CHECKPOINT_DIR \
    --output_folder $OUTPUT_DIR \
    --prompt_file_path $PROMPT_FILE \
    --num_seeds 5
