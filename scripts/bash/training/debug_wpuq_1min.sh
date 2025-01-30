#!/bin/bash
# export TIMEID=$(python scripts/python/init_time_id.py | tail -n 1)
python scripts/python/training/profile_ddpm_pl.py --set_time_id 20250127-1234 \
    --exp_id 0.0.1 \
    --config configs/gpt2-12-s.yaml \
    --dataset wpuq \
    --data_root data/wpuq/ \
    --resolution 1min \
    --train_season whole_year \
    --val_season whole_year \
    --val_every 1 \
    --num_train_step 1 \
    --vectorize_data False \
    --train_batch_size 1