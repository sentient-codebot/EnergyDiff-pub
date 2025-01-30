# !/bin/bash
# ==================================================================================================
# Exp 17.1.6
# Rectified Flow
#   - no rescale 5, learned pos emb
#   - ACTUALLY use 50 steps! 
# Date: 2024-03-19
# Author: Nan Lin

# Get project root directory (regardless of where script is called from)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

python ${PROJECT_ROOT}/scripts/python/training/kernel_rectified_flow.py --exp_id 17.1.6 \
    --config configs/gpt2-12.yaml \
    --dataset cossmic \
    --cossmic_dataset_names "grid-import_residential" \
    --target_labels "[season, area]" \
    --data_root data/cossmic/ \
    --resolution 1min \
    --train_season winter \
    --val_season winter \
    --val_area residential \
    --conditioning False \
    --use_rectified_flow True \
    --num_sampling_step 10 \
    --num_train_step 50000 --save_and_sample_every 50000