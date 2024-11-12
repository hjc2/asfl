import pandas as pd
import numpy as np
from scipy import stats
import argparse
import os
import sys
from tabulate import tabulate  # for pretty printing tables

def analyze_results(file_path):
    """
    Analyze a single results file and return key statistics
    """
    # Read the results file
    df = pd.read_csv(file_path)
    
    # Calculate trend
    slope, intercept, r_value, p_value, std_err = stats.linregress(df['round'], df['accuracy'])
    
    # Return statistics as a dictionary
    return {
        'File': os.path.basename(file_path),
        'Mean Accuracy': df['accuracy'].mean(),
        'Max Accuracy': df['accuracy'].max(),
        'Final Accuracy': df['accuracy'].iloc[-1],
        'Avg Last 5': df['accuracy'].tail(5).mean(),
        'Avg Last 50': df['accuracy'].tail(50).mean(),
        'Learning Rate': slope,
        'Std Dev': df['accuracy'].std(),
        'Median': df['accuracy'].median(),
    }

def analyze_directory(directory):
    """
    Analyze all CSV files in a directory and display results in a sorted table
    """
    # Get all CSV files in directory
    csv_files = [f for f in os.listdir(directory) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in directory: {directory}")
        return
    
    print(f"Found {len(csv_files)} CSV files")
    
    # Analyze all files
    results = []
    for file in csv_files:
        file_path = os.path.join(directory, file)
        stats = analyze_results(file_path)
        results.append(stats)
    
    # Convert to DataFrame for easy sorting and display
    results_df = pd.DataFrame(results)
    
    # Print results sorted by different metrics
    # metrics = ['Mean Accuracy', 'Max Accuracy', 'Final Accuracy', 'Avg Last 5']
    metrics = ['Avg Last 5']
    # metrics = ['Final Accuracy']
    
    for metric in metrics:
        print(f"\nResults sorted by {metric}:")
        print("-" * 100)
        sorted_results = results_df.sort_values(by=metric, ascending=False)
        print(tabulate(sorted_results.round(4), headers='keys', tablefmt='pipe', showindex=False))
        print("\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze ML results from CSV files.')
    parser.add_argument('directory', type=str, help='Directory containing CSV result files')
    args = parser.parse_args()
    
    # Check if directory exists
    if not os.path.isdir(args.directory):
        print(f"Error: Directory '{args.directory}' does not exist")
        sys.exit(1)
    
    analyze_directory(args.directory)

if __name__ == "__main__":
    main()