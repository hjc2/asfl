#!/bin/bash

# Define the archive directory
archive_dir="archive"

# Find the highest version folder number
highest_version=$(ls -d $archive_dir/v* | sed 's/[^0-9]*//g' | sort -n | tail -n 1)

# Determine the next version number
next_version=$((highest_version + 1))

# Create the new version folder
new_folder="$archive_dir/v$next_version"
mkdir "$new_folder"

# Define the log file (adjust if necessary)
log_file="fed_avg.txt"  # Adjust this to the actual log file name if necessary

# Extract configuration information from the log file
num_server_rounds=$(grep -oP 'num server rounds (\d+)' "$log_file" | awk '{print $4}')
local_epochs=$(grep -oP 'num local epochs (\d+)' "$log_file" | awk '{print $4}')
partition=$(grep -oP 'partition: (\w+)' "$log_file" | awk '{print $2}')
fraction=$(grep -oP 'fraction: (\d+)' "$log_file" | awk '{print $2}')

num_supernodes=$((grep -oP 'configure_fit: strategy sampled \d+ clients \(out of \K\d+' "$log_file") | head -n 1)


# Initialize YAML content with server configuration
cat <<EOF > "$new_folder/i.yaml"
server_configuration:
  num_server_rounds: $num_server_rounds
  local_epochs: $local_epochs
  num_supernodes: $num_supernodes
  partition: "$partition"
  fraction: $fraction
EOF

# Find and move all fed_*.txt files from outside the archive directory to the new folder
fed_files=$(ls fed_*.txt 2>/dev/null)

if [[ -z "$fed_files" ]]; then
    echo "No fed_*.txt files found to move outside the archive."
else
    # Move the files to the new folder
    mv fed_*.txt "$new_folder"
    echo "Moved fed_*.txt files to $new_folder."
fi

# Run ./scripts/many.sh on the new folder
./scripts/many.sh "$new_folder"

# Done
echo "Done! Created i.yaml and executed ./scripts/many.sh on $new_folder."


git add .

git commit -m "new run $new_folder"
git push

echo "All operations completed successfully!"

