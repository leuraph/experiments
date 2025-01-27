#!/bin/bash

# Ensure the script is called with one argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <path_to_exp>"
    exit 1
fi

# Get the argument
path_to_exp="$1"

# Check if the argument is a valid directory
if [ ! -d "$path_to_exp" ]; then
    echo "Error: $path_to_exp is not a valid directory."
    exit 1
fi

# Extract the experiment name (last part of the path)
experiment_name=$(basename "$path_to_exp")

# Loop through all directories in path_to_exp
for dir in "$path_to_exp"/*/; do
    if [ -d "$dir" ]; then
        # Extract the name of the current directory
        dir_name=$(basename "$dir")

        # Create the output file name dynamically
        output_file="plots/${experiment_name}_${dir_name}.pdf"

        # Call the Python script with the specified arguments
        python plot_energy_norm_errors_for_exp_03.py --path "$dir" -o "$output_file"
    fi
done
