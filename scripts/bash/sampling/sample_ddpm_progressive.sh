# !/bin/bash
# ==================================================================================================
# Exp 13.0.0
# grid_import_residential
# Training from scratch, cossmic data. grid import residential
#   - training with same config as 9.0.0. reverted most script changes. 
#   - old attention. 
#   - dpm solver sampling
# Date: 2024-02-16
# Author: Nan Lin
# GPU memory: 

# Get project root directory (regardless of where script is called from)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

python ${PROJECT_ROOT}/scripts/python/sampling/sample_ddpm_progressive.py --exp_id 13.0.0 \
    --config configs/gpt2-12.yaml \
    --dataset cossmic \
    --cossmic_dataset_names "grid-import_residential" \
    --target_labels "[season, area]" \
    --data_root data/cossmic/ \
    --resolution 1min \
    --vectorize_data True \
    --style_vectorize patchify \
    --vectorize_window_size 8 \
    --train_season winter \
    --val_season winter \
    --val_area residential \
    --conditioning False \
    --learn_variance False \
    --sigma_small True \
    --train_batch_size 64 \
    --train_lr 0.0001 \
    --num_sample 2000 \
    --val_every 1250 \
    --val_batch_size 100 \
    --num_sampling_step 1000 \
    --num_diffusion_step 1000 \
    --num_train_step 100000 --save_and_sample_every 50000 \
    --load_runid 20240222-8654 \
    --load_milestone 2