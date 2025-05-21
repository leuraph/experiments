#!/bin/bash

# USAGE
# -----
#
# bash calculate_all_errors.sh <script_path> <folder_path> <energy_file_path>
# where
# <script_path>: python script that calculates the errors
# <folder_path>: folder to a certain experiment
# <energy_file_path>: file containing a reference value of the exact energy norm squared

# Check for exactly 3 arguments
if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <script_path> <folder_path> <energy_file_path>"
    exit 1
fi

SCRIPT_PATH="$1"
FOLDER_PATH="$2"
ENERGY_FILE="$3"

# Check that script and energy file exist
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "Error: Script '$SCRIPT_PATH' not found!"
    exit 1
fi

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
        python "$SCRIPT_PATH" --path "$dir" --energy-path "$ENERGY_FILE"
    fi
done