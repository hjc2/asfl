import pandas as pd
import sys
import os
from pathlib import Path

def process_directory(directory_path):
    # Ensure the directory path exists
    if not os.path.exists(directory_path):
        print(f"Error: Directory '{directory_path}' not found")
        return

    # Get all CSV files in the directory
    csv_files = list(Path(directory_path).glob('*.csv'))
    
    if not csv_files:
        print(f"No CSV files found in {directory_path}")
        return

    # Process each CSV file
    for csv_path in csv_files:
        try:
            df = pd.read_csv(csv_path)
            total_count = df['count'].sum()
            print(f"{csv_path.name}: Total count = {total_count}")
        except Exception as e:
            print(f"Error processing {csv_path.name}: {str(e)}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python reader.py <directory_path>")
        sys.exit(1)
    
    directory_path = sys.argv[1]
    process_directory(directory_path)