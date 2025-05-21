#!/bin/bash

# USAGE
# -----
#
# bash calculate_all_errors.sh <folder_path> <problem_number> <energy_file_path>
# where
# <folder_path>: folder to a certain experiment
# <problem_number>: number of the problem under consideration
# <energy_file_path>: file containing a reference value of the exact energy norm squared

# Check for exactly 3 arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <folder_path> <problem_number> <energy_file_path>"
    exit 1
fi

FOLDER_PATH="$1"
PROBLEM_NUMBER="$2"
ENERGY_FILE="$3"

if [ ! -f "$ENERGY_FILE" ]; then
    echo "Error: Energy file '$ENERGY_FILE' not found!"
    exit 1
fi

# Check that folder exists
if [ ! -d "$FOLDER_PATH" ]; then
    echo "Error: Folder '$FOLDER_PATH' not found!"
    exit 1
fi

# Iterate through subdirectories
for dir in "$FOLDER_PATH"/*/; do
    if [ -d "$dir" ]; then
        echo "Calling script for directory: $dir"
        python calc_errors_for_problem.py --number "$PROBLEM_NUMBER" --path "$dir" --energy-path "$ENERGY_FILE"
    fi
done