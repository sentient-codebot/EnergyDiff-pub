#!/bin/bash

BLUE="\033[34m"
NC="\033[0m"

# Define the path to the temporary file
temp_file="/tmp/run_id_store_calibrate_192andfo91.txt"

# Check if the temporary file exists and read from it if it does
if [ -f "$temp_file" ]; then
    default_run_id=$(<"$temp_file")
else
    default_run_id=""
fi

# Prompt the user for run_id with the default value suggested
echo -ne "${BLUE}"
read -p "runid [${default_run_id}]: " run_id
echo -ne "${NC}"

# If the user did not provide a new run_id, use the default one
if [ -z "$run_id" ]; then
    run_id=$default_run_id
fi

# Save the new or unchanged run_id back to the file
echo "$run_id" > "$temp_file"
echo -e "\033[34mUsing run_id: $run_id\033[0m"


# Get project root directory (regardless of where script is called from)
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../" && pwd)"

python ${PROJECT_ROOT}/scripts/python/postprocessing/do_calibrate.py --load_runid $run_id \
    --load_milestone 1 \
    --num_sampling_step 100