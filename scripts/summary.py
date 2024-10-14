

import sys

import re

import ast
import csv


def process_large_file(input_filename, output_filename):
    """
    Process a large text file, extracting lines starting from the line that contains "[SUMMARY]".
    
    :param input_filename: Path to the input file.
    :param output_filename: Path to the output file.
    """
    found_summary = False

    pattern = re.compile(r'^dv - \| INFO flwr \d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3} \| server\.py:\d+ \| \t*')

    with open(input_filename, 'r') as infile, open(output_filename, 'w') as outfile:
        for line in infile:
            if not found_summary:
                if 'History (metrics, distributed, evaluate):' in line:
                    found_summary = True
                    # # Write the line containing [SUMMARY] and all subsequent lines to the output file
                    # outfile.write(line)
            elif 'DEBUG' in line:
                pass
            else:
                # Write all subsequent lines after finding [SUMMARY]

                cleaned_line = pattern.sub('', line)

                outfile.write(cleaned_line)

def load_dict_from_txt(filename):
    """
    Load a dictionary from a text file where each line is a key-value pair separated by a colon.
    
    :param filename: Path to the text file.
    :return: A dictionary with key-value pairs.
    """
    with open(filename, 'r') as file:
        # Read the entire file content
        file_content = file.read().strip()
        
        # Convert the string representation of the dictionary to an actual dictionary
        data_dict = ast.literal_eval(file_content)
    
    return data_dict

def merge_nested_dict(nested_dict):
    result = {}
    
    # Get all unique keys (e.g., 1, 2, 3, 4, 5)
    all_keys = set()
    for field_data in nested_dict.values():
        all_keys.update(dict(field_data).keys())
    
    # Merge the data
    for key in all_keys:
        result[key] = {}
        for field, data in nested_dict.items():
            dict_data = dict(data)
            if key in dict_data:
                result[key][field] = dict_data[key]
    
    return result

def write_dict_to_csv(data, filename='output.csv'):
    if not data:
        raise ValueError("The input dictionary is empty")

    # Dynamically get all unique keys (column names)
    fieldnames = set()
    for values in data.values():
        fieldnames.update(values.keys())
    
    # Ensure 'name' is the first column
    fieldnames = ['round'] + sorted(list(fieldnames))

    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        
        # Write the header
        writer.writeheader()
        
        # Write the data
        for key, values in data.items():
            row = {'round': key}  # Start with the 'name' column
            row.update(values)  # Add all other columns
            writer.writerow(row)

def main():
    if len(sys.argv) != 2:
        print("Usage: python summary.py <input_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = input_file.split('.')[0] + "-out"

    process_large_file(input_file, output_file + '.txt')

    dictionary = load_dict_from_txt(output_file + '.txt')
    merged_dict = merge_nested_dict(dictionary)

    write_dict_to_csv(merged_dict, output_file + '.csv')

if __name__ == "__main__":
    main()
