# !/bin/bash
# ==================================================================================================
# Exp 21.1.0
# Diffusion
#   - lcl 1h
# Date: 2024-03-28
# Author: Nan Lin

# Get project root directory (regardless of where script is called from)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

accelerate launch ${PROJECT_ROOT}/scripts/python/training/train_ddpm_conditional.py --exp_id 21.1.0 \
    --config configs/gpt2-12.yaml \
    --vectorize_data False \
    --dataset lcl_electricity \
    --data_root data/lcl_electricity/ \
    --resolution 1h \
    --use_rectified_flow False \
    --num_sampling_step 50 \
    --num_train_step 50000 --save_and_sample_every 50000