#!/bin/bash

# Counter for how many consecutive minutes GPU 7 is idle
idle_count=0

# Threshold for idleness in minutes
threshold=15

while true; do
    # Get all GPU utilizations using nvidia-smi
    utils=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)
    
    echo "$(date): All GPU Utilizations = $utils"

    # Check if all GPUs are idle (utilization = 0)
    all_idle=true
    while IFS= read -r util; do
        if [ "$util" -ne 0 ]; then
            all_idle=false
            break
        fi
    done <<< "$utils"

    if [ "$all_idle" = true ]; then
        ((idle_count++))
        echo "All GPUs have been idle for $idle_count minute(s)."
    else
        idle_count=0
    fi

    if [ "$idle_count" -ge "$threshold" ]; then
        echo "All GPUs have been idle for $threshold minutes. Running occupy.py..."
        python ./azure/occupy.py
        break
    fi

    sleep 60
done