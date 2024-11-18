#!/bin/bash

# Configuration

base_name="pltResults"    # Base filename

start_num=200       # Starting number
end_num=33000       # Ending number
increment=200       # Increment step
extension2="csv"    # File extension

# Function to sort a single file
tke_for_file() {
    local file="$1"

    # Check if the file exists
    if [ ! -f "$file" ]; then
        echo "Warning: File '$file' not found, skipping..."
        return 1
    fi

    # Check if the file is empty
    if [ ! -s "$file" ]; then
        echo "Warning: File '$file' is empty, skipping..."
        return 1
    fi

    # Create a temporary file
    local temp_file=$(mktemp)

    # Sort the CSV file based on the first column numerically in descending order
    sort -t',' -k1,1nr "$file" > "$temp_file"

    # Replace the original file with the sorted version
    mv "$temp_file" "$file"
    echo "Sorted: $file"

    return 0

}

# Initialize counters
total_files=0
processed_files=0
failed_files=0

# Calculate expected number of files
expected_files=$(( (end_num - start_num) / increment + 1 ))
echo "Will process files with numbers: "

for ((i=start_num; i<=end_num; i+=increment)); do
    echo -n "$i "
done
echo -e "\n"

# Process each file in the sequence
for ((i=start_num; i<=end_num; i+=increment)); do
    # Construct the filename
    if [ $i -lt 999 ]; then
        prefix="00"
    elif [ $i -gt 1000 && $i -lt 9999 ]; then
        prefix="0"
    else
        prefix=""
    fi
    iter_file="${base_name}${prefix}${i}.${extension}"

    total_files=$((total_files + 1))
    echo "Processing file $total_files of $expected_files: $total_file"

    if tke_for_file "$current_file"; then
        processed_files=$((processed_files + 1))
    else
        failed_files=$((failed_files + 1))
    fi
done

# Print summary
echo "================="
echo "Process Complete!"
echo "================="
echo "Total files: $total_files"
echo "Successfully processed: $processed_files"
echo "Failed/Skipped: $failed_files"
