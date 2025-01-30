#!/bin/bash

# Get project root directory (regardless of where script is called from)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

python ${PROJECT_ROOT}/scripts/python/evaluation/create_umap_mapper.py --exp_id 0.1.0 \
    --config configs/gpt2-12.yaml \
    --dataset cossmic \
    --cossmic_dataset_names "grid-import_residential" \
    --data_root data/cossmic/ \
    --target_labels "[season, area]" \
    --train_season whole_year \
    --val_season whole_year \
    --val_area residential \
    --resolution 1min \
    --vectorize_data False