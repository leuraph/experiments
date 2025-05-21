#!/bin/bash

# USAGE
# -----
#
# bash calculate_all_errors.sh <script_path> <folder_path> <energy_file_path>
# where
# <script_path>: python script that plots the errors
# <folder_path>: folder to a certain experiment

# Check for exactly 3 arguments
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <script_path> <folder_path>"
    exit 1
fi

SCRIPT_PATH="$1"
FOLDER_PATH="$2"

# Check that script and energy file exist
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script '$SCRIPT_PATH' not found!"
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
        OUTPUT_PATH="${dir/results/plots}"
        OUTPUT_PATH="${OUTPUT_PATH%/}.pdf"
        python "$SCRIPT_PATH" --path "$dir" -o "$OUTPUT_PATH"
    fi
done