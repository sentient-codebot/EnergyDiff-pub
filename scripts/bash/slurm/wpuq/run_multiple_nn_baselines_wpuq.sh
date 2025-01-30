#!/bin/bash

#SBATCH --job-name="nb_w_heat"
#SBATCH --partition=gpu
#SBATCH --time=5:00:00
#SBATCH --gpus=1
#SBATCH --output=slurm_%j.log
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=

module load 2023
module load Miniconda3/23.5.2-0

unset CONDA_SHLVL
source "$(conda info --base)/etc/profile.d/conda.sh"

conda activate energydiff
cd ~/projects/HeatoDiff

# Define array of TIMEIDs - you can add more IDs here
TIMEIDS=(
    "20250117-4505"
)

# Loop through each TIMEID
for TIMEID in "${TIMEIDS[@]}"; do
    echo "**************** [TIMEID: $TIMEID] job started.           **************************"

    # train
    # srun python scripts/python/training/train_vae_pl.py --load_time_id $TIMEID
    # echo "**************** [TIMEID: $TIMEID] vae training completed.**************************"

    srun python scripts/python/training/train_gan_pl.py --load_time_id $TIMEID
    echo "**************** [TIMEID: $TIMEID] gan training completed.**************************"

    # sample vae, gan
    # srun python scripts/python/sampling/sample_vae_pl.py --load_time_id $TIMEID
    srun python scripts/python/sampling/sample_gan_pl.py --load_time_id $TIMEID
    echo "**************** [TIMEID: $TIMEID] nn baseline completed. **************************"

    # # sample gmm, copula, ddpm
    # timeout 1h srun python scripts/python/sampling/sample_baseline.py --load_time_id $TIMEID
    # echo "**************** [TIMEID: $TIMEID] gmm/copula completed.  **************************"
    # srun python scripts/python/sampling/sample_ddpm_pl.py --load_time_id $TIMEID \
    #     --num_sample 4000 \
    #     --num_sampling_step 100

    # test
    srun python scripts/python/evaluation/test_metrics_pl.py --load_time_id $TIMEID
    echo "**************** [TIMEID: $TIMEID] test completed.        **************************"
    
    echo "**************** [TIMEID: $TIMEID] all tasks completed.   **************************"
    echo ""
done