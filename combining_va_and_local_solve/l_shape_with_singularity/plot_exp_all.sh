#!/bin/bash

# Get the argument
path_to_exp="results/experiment_02"

# Check if the argument is a valid directory
if [ ! -d "$path_to_exp" ]; then
    echo "Error: $path_to_exp is not a valid directory."
    exit 1
fi

# Extract the experiment name (last part of the path)
experiment_name=$(basename "$path_to_exp")

# Get the list of directories
dirs=("$path_to_exp"/*/)
total_dirs=${#dirs[@]}
current_dir=0

# Loop through all directories in path_to_exp
for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        # Extract the name of the current directory
        dir_name=$(basename "$dir")

        # Create the output file name dynamically
        output_file="plots/${experiment_name}_${dir_name}.pdf"

        # Call the Python script with the specified arguments
        python plot_energy_errors_exp.py --path "$dir" -o "$output_file"

        # Update progress
        current_dir=$((current_dir + 1))
        progress=$((current_dir * 100 / total_dirs))
        printf "\rProgress: [%-100s] %d%%" $(printf "#%.0s" $(seq 1 $progress)) $progress
    fi
done

# Print a newline after the progress bar
echo