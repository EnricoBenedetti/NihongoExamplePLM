#!/bin/bash

# Check if exactly two arguments are given (source file and destination directory)
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <source_file> <destination_directory>"
    exit 1
fi

# Assign arguments to variables
source_file="$1"
destination_directory="$2"
destination_file="${destination_directory}/$(basename "$source_file")"

# Ensure the destination directory exists
if [ ! -d "$destination_directory" ]; then
    echo "The destination directory '$destination_directory' does not exist."
    exit 2
fi

# Initialize variables
copy=false

# Read the source file line by line
while IFS= read -r line; do
    # Check for the start separator
    if [[ $line == '===========RESPONSE===========' ]]; then
        copy=true
        continue
    fi
    # Check for the end separator
    if [[ $line == "+++++END_RESPONSE+++++" ]]; then
        copy=false
        continue
    fi
    # Copy the line if it's between the start and end separators
    if $copy; then
        echo "$line" >> "$destination_file"
    fi
done < "$source_file"

