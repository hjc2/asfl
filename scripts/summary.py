

import sys

import re

import ast


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

    

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_file> <output_file>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    process_large_file(input_file, output_file)

    dictionary = load_dict_from_txt(output_file)
    # print(dictionary['accuracy'][0])

    print(dictionary)
    # for x in dictionary:
    #     print(dictionary[x])

if __name__ == "__main__":
    main()
