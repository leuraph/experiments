#!/bin/bash

RESULTS_DIR="results"

for folder in "$RESULTS_DIR"/problem-*; do
    if [ -d "$folder" ]; then
        # Extract problem number using parameter expansion and regex
        folder_name=$(basename "$folder")
        if [[ $folder_name =~ problem-([0-9]+)_ ]]; then
            problem_num="${BASH_REMATCH[1]}"
            if [[ "$problem_num" == "4" ]]; then
                python plot_energy_norm_errors.py \
                    --path "$folder"
            fi
        fi
    fi
done