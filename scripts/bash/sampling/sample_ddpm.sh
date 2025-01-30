# !/bin/bash
# ==================================================================================================
# Exp 22.0.0
#   - cossmic grid
# Date: 2024-03-28
# Author: Nan Lin

# Get project root directory (regardless of where script is called from)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

accelerate launch ${PROJECT_ROOT}/scripts/python/sampling/dpm-solver-sample.py --load_runid 20240402-5312 \
    --load_milestone 1 \
    --val_batch_size 500 \
    --num_sample 4000 \
    --num_sampling_step 100 \
    --dpm_solver_sample True