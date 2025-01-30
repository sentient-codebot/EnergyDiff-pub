#!/bin/bash

#SBATCH --job-name="t_wpuq"
#SBATCH --partition=gpu
#SBATCH --time=2:00:00
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
    "20250116-5446"
    "20250116-3200"
    "20250118-9946"
)

# Loop through each TIMEID
for TIMEID in "${TIMEIDS[@]}"; do
    echo "**************** [TIMEID: $TIMEID] job started.           **************************"

    # test
    srun python scripts/python/evaluation/test_metrics_ref.py --load_time_id $TIMEID
    echo "**************** [TIMEID: $TIMEID] test completed.        **************************"
    
    echo "**************** [TIMEID: $TIMEID] all tasks completed.   **************************"
    echo ""
done