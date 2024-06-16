#!/bin/bash

# Set the directory containing the checkpoints
checkpoint_dir="/home/csgrad/sahmed9/reps/RealCQA/code/outputChartConv0"

# Create an array to hold all checkpoint files
ckp_files=("$checkpoint_dir"/*.pt)

# Determine the number of checkpoint files
num_ckp_files=${#ckp_files[@]}

# Calculate the number of files per batch (assuming 4 parallel jobs)
batch_size=$((num_ckp_files / 1))

# Function to run evaluations on a subset of checkpoint files
run_evaluations() {
    local start_index=$1
    local end_index=$2
    for (( i=start_index; i<end_index; i++ )); do
        if [[ -f ${ckp_files[i]} ]]; then
            echo "Running evaluation for checkpoint: ${ckp_files[i]}"
            python /home/csgrad/sahmed9/reps/RealCQA/code/chartconv_matcha_eval.py --ckpt_path "${ckp_files[i]}"
            # Check if a kill signal was received
            if [[ "$killed" == "true" ]]; then
                echo "Exiting due to interrupt signal."
                exit 1
            fi
        fi
    done
}

# Function to handle interrupt signal
handle_kill() {
    echo "Received kill signal, cleaning up..."
    killed="true"
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap 'handle_kill' SIGINT SIGTERM

# Start parallel evaluations with staggering
killed="false"
for (( i=0; i<4; i++ )); do
    start_index=$((i * batch_size))
    end_index=$(((i + 1) * batch_size))
    # For the last batch, make sure to include all remaining files
    if [[ $i -eq 3 ]]; then
        end_index=$num_ckp_files
    fi
    run_evaluations $start_index $end_index &
    sleep 30
done

# Wait for all parallel jobs to finish
wait

echo "All evaluations completed."
