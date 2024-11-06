
#!/bin/bash

# Function to get the directory of the bash script
get_script_dir() {
    SOURCE="${BASH_SOURCE[0]}"
    while [ -h "$SOURCE" ]; do
        DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
        SOURCE="$(readlink "$SOURCE")"
        [[ $SOURCE != /* ]] && SOURCE="$DIR/$SOURCE"
    done
    DIR="$( cd -P "$( dirname "$SOURCE" )" && pwd )"
    echo "$DIR"
}

# Get the directory where the bash script is located
SCRIPT_DIR=$(get_script_dir)

# Check if a directory path is provided
if [ $# -eq 0 ]; then
    echo "Usage: $0 <directory_path>"
    exit 1
fi

# Get the directory path from the command line argument
directory="$1"

# Check if the directory exists
if [ ! -d "$directory" ]; then
    echo "Error: Directory '$directory' does not exist."
    exit 1
fi

# Check if the Python script exists in the same directory as this bash script
if [ ! -f "$SCRIPT_DIR/summary.py" ]; then
    echo "Error: summary.py not found in the script directory: $SCRIPT_DIR"
    exit 1
fi

# Process all .txt files in the specified directory
for file in "$directory"/*.txt; do
    if [ -f "$file" ]; then
        echo "Processing file: $file"
        python "$SCRIPT_DIR/summary.py" "$file"
        
        # Check if the Python script execution was successful
        if [ $? -eq 0 ]; then
            echo "Successfully processed $file"
        else
            echo "Error processing $file"
        fi
        
        echo "------------------------"
    fi
done

echo "All .txt files have been processed."
