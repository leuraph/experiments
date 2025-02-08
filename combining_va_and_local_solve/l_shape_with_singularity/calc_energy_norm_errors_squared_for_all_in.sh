#!/bin/bash

# Ensure the script is called with two arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <path_to_exp_dir> <path_to_energy>"
    exit 1
fi

# Get arguments
path_to_exp_dir="$1"
path_to_energy="$2"

# Check if the first argument is a directory
if [ ! -d "$path_to_exp_dir" ]; then
    echo "Error: $path_to_exp_dir is not a valid directory."
    exit 1
fi

# Check if the second argument is a file
if [ ! -f "$path_to_energy" ]; then
    echo "Error: $path_to_energy is not a valid file."
    exit 1
fi

# Loop through each directory in path_to_exp_dir
for dir in "$path_to_exp_dir"/*/; do
    if [ -d "$dir" ]; then
        # Call the Python script with the specified arguments
        python calc_errors_for_exp.py --path "$dir" --energy-path "$path_to_energy"
    fi
done
