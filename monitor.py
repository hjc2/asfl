import os
from pathlib import Path
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Tuple

class LogAnalyzer:
    # Compile regex patterns once for better performance
    ACCURACY_PATTERN = re.compile(r'accuracy: ([\d.]+)')
    ROUND_PATTERN = re.compile(r'\[ROUND (\d+)\]')
    
    @staticmethod
    def get_last_values(file_path: str) -> Tuple[str, int, float]:
        """Get the most recent round and accuracy from a single file."""
        filename = Path(file_path).name
        last_round = None
        last_accuracy = None
        
        # Read last 4KB of file (adjust if needed)
        with open(file_path, 'rb') as file:
            # Jump to end of file
            file.seek(0, 2)
            file_size = file.tell()
            
            # Read last chunk of file
            chunk_size = min(4096, file_size)
            file.seek(file_size - chunk_size)
            chunk = file.read(chunk_size).decode()
            
            # Split into lines and reverse for efficient searching
            lines = chunk.split('\n')
            
            # Find last accuracy
            for line in reversed(lines):
                if last_accuracy is None and 'aggregated accuracy:' in line:
                    match = LogAnalyzer.ACCURACY_PATTERN.search(line)
                    if match:
                        last_accuracy = float(match.group(1))
                        continue
                
                if last_round is None and '[ROUND' in line:
                    match = LogAnalyzer.ROUND_PATTERN.search(line)
                    if match:
                        last_round = int(match.group(1))
                        break
        
        return filename, last_round, last_accuracy

    @staticmethod
    def process_directory(directory: str, pattern: str = "*.txt") -> None:
        """Process all matching files in directory and print results."""
        directory_path = Path(directory)
        files = list(directory_path.glob(pattern))
        
        # Process files in parallel
        with ThreadPoolExecutor() as executor:
            results = list(executor.map(LogAnalyzer.get_last_values, files))
        
        # Print results in a clean format
        print("\nMost Recent Values:")
        print("-" * 60)
        print(f"{'Filename':<30} {'Round':>8} {'Accuracy':>12}")
        print("-" * 60)
        
        for filename, round_num, accuracy in sorted(results):
            if round_num is not None and accuracy is not None:
                print(f"{filename:<30} {round_num:>8d} {accuracy:>12.4f}")
            else:
                print(f"{filename:<30} {'N/A':>8} {'N/A':>12}")

def main():
    # Example usage
    analyzer = LogAnalyzer()
    analyzer.process_directory('')

if __name__ == "__main__":
    main()