#!/bin/bash

# This script computes the energies for all the results

RESULTS_DIR="results"

echo "Computing energies for problem 1"
for folder in "$RESULTS_DIR"/problem-*; do
    if [ -d "$folder" ]; then
        # Extract problem number using parameter expansion and regex
        folder_name=$(basename "$folder")
        if [[ $folder_name =~ problem-([0-9]+)_ ]]; then
            problem_num="${BASH_REMATCH[1]}"
            if [[ "$problem_num" == "1" ]]; then
                python compute_energies.py --problem "$problem_num" --path "$folder"
            fi
        fi
    fi
done

echo "Computing energies for problem 2"
for folder in "$RESULTS_DIR"/problem-*; do
    if [ -d "$folder" ]; then
        # Extract problem number using parameter expansion and regex
        folder_name=$(basename "$folder")
        if [[ $folder_name =~ problem-([0-9]+)_ ]]; then
            problem_num="${BASH_REMATCH[1]}"
            if [[ "$problem_num" == "2" ]]; then
                python compute_energies.py --problem "$problem_num" --path "$folder"
            fi
        fi
    fi
done

echo "Computing energies for problem 3"
for folder in "$RESULTS_DIR"/problem-*; do
    if [ -d "$folder" ]; then
        # Extract problem number using parameter expansion and regex
        folder_name=$(basename "$folder")
        if [[ $folder_name =~ problem-([0-9]+)_ ]]; then
            problem_num="${BASH_REMATCH[1]}"
            if [[ "$problem_num" == "3" ]]; then
                python compute_energies.py --problem "$problem_num" --path "$folder"
            fi
        fi
    fi
done