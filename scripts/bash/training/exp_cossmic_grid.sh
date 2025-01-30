# !/bin/bash
# ==================================================================================================
# Exp 22.0.0
#   - cossmic grid
# Date: 2024-03-28
# Author: Nan Lin

# Get project root directory (regardless of where script is called from)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

accelerate launch ${PROJECT_ROOT}/scripts/python/training/train_ddpm_conditional.py --exp_id 22.0.0 \
    --config configs/gpt2-12.yaml \
    --dataset cossmic \
    --cossmic_dataset_names "grid-import_residential" \
    --target_labels "[season, area]" \
    --data_root data/cossmic/ \
    --resolution 1min \
    --train_season whole_year \
    --val_season whole_year \
    --val_area residential \
    --conditioning False \
    --use_rectified_flow False \
    --num_sampling_step 50 \
    --num_train_step 50000 --save_and_sample_every 50000