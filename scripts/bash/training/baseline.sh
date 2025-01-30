read -p "runid:" runid

# Get project root directory (regardless of where script is called from)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

python3 ${PROJECT_ROOT}/scripts/python/training/train_gmm_one_season.py --load_runid $runid \
    --load_milestone 1 \
    --num_sample 4000 

python3 ${PROJECT_ROOT}/scripts/python/training/train_copula_one_season.py --load_runid $runid \
    --load_milestone 1 \
    --num_sample 4000 