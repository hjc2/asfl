import os
from pathlib import Path
import re
import matplotlib.pyplot as plt
import pandas as pd
from typing import Dict, List, Tuple
import concurrent.futures

class LogAnalyzer:
    # Compile regex patterns once for better performance
    ACCURACY_PATTERN = re.compile(r'accuracy: ([\d.]+)')
    ROUND_PATTERN = re.compile(r'\[ROUND (\d+)\]')
    
    @staticmethod
    def process_single_file(file_path: str) -> Dict[int, float]:
        """Process a single log file and return round:accuracy mapping."""
        results = {}
        
        # Read file from bottom up using seek
        with open(file_path, 'rb') as file:
            # Jump to end of file
            file.seek(0, 2)
            file_size = file.tell()
            
            # Initialize variables for reading backwards
            block_size = 8192
            last_line = ""
            current_position = file_size
            
            while current_position > 0:
                # Calculate new position
                new_position = max(current_position - block_size, 0)
                file.seek(new_position)
                block = file.read(current_position - new_position).decode()
                
                # Add the last incomplete line
                block += last_line
                
                # Split into lines
                lines = block.split('\n')
                
                # If we're not at the start, first line is incomplete
                if new_position > 0:
                    last_line = lines[0]
                    lines = lines[1:]
                else:
                    last_line = ""
                
                # Process each line
                for line in reversed(lines):
                    if 'aggregated accuracy:' in line:
                        accuracy_match = LogAnalyzer.ACCURACY_PATTERN.search(line)
                        if accuracy_match:
                            accuracy = float(accuracy_match.group(1))
                            
                            # Find the corresponding round number
                            round_lines = [l for l in reversed(lines) if '[ROUND' in l]
                            if round_lines:
                                round_match = LogAnalyzer.ROUND_PATTERN.search(round_lines[0])
                                if round_match:
                                    round_num = int(round_match.group(1))
                                    results[round_num] = accuracy
                
                current_position = new_position
                
                # Break if we've found enough data points
                if len(results) >= 10:  # Adjust this number based on your needs
                    break
                    
        return results

    @staticmethod
    def process_multiple_files(directory: str, pattern: str = "*.log") -> pd.DataFrame:
        """Process multiple log files in parallel and return a DataFrame."""
        directory_path = Path(directory)
        files = list(directory_path.glob(pattern))
        
        # Process files in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_to_file = {
                executor.submit(LogAnalyzer.process_single_file, str(file_path)): file_path
                for file_path in files
            }
            
            results = {}
            for future in concurrent.futures.as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    file_results = future.result()
                    results[file_path.stem] = file_results
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
        
        # Convert to DataFrame
        df = pd.DataFrame.from_dict(results, orient='index')
        return df

    @staticmethod
    def plot_results(df: pd.DataFrame, output_path: str = 'accuracy_trends.png'):
        """Plot accuracy trends for all processed files."""
        plt.figure(figsize=(12, 6))
        for file_name in df.index:
            plt.plot(df.columns, df.loc[file_name], label=file_name, marker='o')
        
        plt.xlabel('Round Number')
        plt.ylabel('Accuracy')
        plt.title('Accuracy Trends Across Files')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

def main():
    # Example usage
    analyzer = LogAnalyzer()
    
    # Process all log files in a directory
    results_df = analyzer.process_multiple_files('', '*.txt')
    
    # Print the results
    print("\nResults Summary:")
    print(results_df)
    
    # Plot the results
    analyzer.plot_results(results_df)

if __name__ == "__main__":
    main()