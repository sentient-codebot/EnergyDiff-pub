# !/bin/bash
# ==================================================================================================
# Exp 23.0.0
# wpuq trafo
#   - wpuq 1h
# Date: 2024-04-02
# Author: Nan Lin

# Get project root directory (regardless of where script is called from)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

accelerate launch ${PROJECT_ROOT}/scripts/python/training/train_ddpm_conditional.py --exp_id 23.0.0 \
    --config configs/gpt2-12.yaml \
    --dataset wpuq_trafo \
    --data_root data/wpuq/ \
    --resolution 1min \
    --vectorize_data True \
    --train_season winter \
    --val_season winter \
    --val_area residential \
    --conditioning False \
    --use_rectified_flow False \
    --num_sampling_step 50 \
    --num_train_step 50000 --save_and_sample_every 50000