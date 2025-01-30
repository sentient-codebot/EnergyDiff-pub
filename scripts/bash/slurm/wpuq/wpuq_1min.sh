#!/bin/bash

#SBATCH --job-name=
#SBATCH --partition=gpu
#SBATCH --time=13:00:00
#SBATCH --gpus=2
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=

module load 2023
module load Miniconda3/23.5.2-0

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate energydiff
cd ~/projects/HeatoDiff

# init time id for all tasks
export TIMEID=$(python scripts/python/init_time_id.py | tail -n 1)
echo "**************** [TIMEID: $TIMEID] job started.           **************************"
# train/sample/test
srun python scripts/python/training/train_ddpm_pl.py --set_time_id $TIMEID \
    --exp_id 1.0.0 \
    --config configs/gpt2-12.yaml \
    --dataset wpuq \
    --data_root data/wpuq/ \
    --resolution 1min \
    --train_season whole_year \
    --val_season whole_year
echo "**************** [TIMEID: $TIMEID] training completed.    **************************"


# train baseline
timeout 2h srun python scripts/python/training/train_vae_pl.py --load_time_id $TIMEID
echo "**************** [TIMEID: $TIMEID] vae training completed.**************************"
timeout 2h srun python scripts/python/training/train_gan_pl.py --load_time_id $TIMEID
echo "**************** [TIMEID: $TIMEID] gan training completed.**************************"


# sample
timeout 1h srun python scripts/python/sampling/sample_baseline.py --load_time_id $TIMEID
echo "**************** [TIMEID: $TIMEID] gmm/copula completed.  **************************"
srun python scripts/python/sampling/sample_ddpm_pl.py --load_time_id $TIMEID \
    --num_sample 4000 \
    --num_sampling_step 100
echo "**************** [TIMEID: $TIMEID] ddpm completed.        **************************"
srun python scripts/python/sampling/sample_vae_pl.py --load_time_id $TIMEID
srun python scripts/python/sampling/sample_gan_pl.py --load_time_id $TIMEID
echo "**************** [TIMEID: $TIMEID] nn baseline completed. **************************"


# test
srun python scripts/python/evaluation/test_metrics_pl.py --load_time_id $TIMEID
echo "**************** [TIMEID: $TIMEID] test completed.        **************************"