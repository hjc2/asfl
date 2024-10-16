import pandas as pd
import matplotlib.pyplot as plt
import sys
import yaml
import os
import glob

def round_to_nearest_5_or_10(n):
    nearest_5 = round(n / 5) * 5
    nearest_10 = round(n / 10) * 10
    if abs(n - nearest_5) <= abs(n - nearest_10):
        return nearest_5
    else:
        return nearest_10

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <FileName> <Directory>")
        sys.exit(1)

    directory = sys.argv[2]
    dir = sys.argv[2]
    reference_file = directory + sys.argv[1]
    config_file = directory + 'i.yaml'

    # Load YAML configuration
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Extracting configuration information
    num_server_rounds = config['server_configuration']['num_server_rounds']
    local_epochs = config['server_configuration']['local_epochs']
    num_supernodes = config['server_configuration']['num_supernodes']
    partitioner = config['server_configuration']['partition']

    reference_data = pd.read_csv(reference_file)
    csv_files = glob.glob(os.path.join(directory, '*.csv'))

    fig, ax = plt.subplots(figsize=(12, 8))

    # Plot the accuracy difference for each CSV file
    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        
        print(csv_file)
        print(reference_file)
        if os.path.normpath(csv_file) == os.path.normpath(reference_file):
            continue            
            print("activated")
            print(csv_file)
            continue

        # Ensure both DataFrames have the same number of rounds
        min_rounds = min(len(reference_data), len(data))
        reference_data_trimmed = reference_data.head(min_rounds)
        data_trimmed = data.head(min_rounds)

        # Calculate the difference in accuracy
        accuracy_diff = data_trimmed['accuracy'] - reference_data_trimmed['accuracy']

        # Plot the accuracy difference
        ax.plot(data_trimmed['round'], accuracy_diff, linestyle='-', label=os.path.basename(csv_file))

    # Adding title and labels
    # ax.set_title('Accuracy Difference by Round : ' + directory)
    plt.title(f"Partitioner: {partitioner}", fontsize=16)
    plt.suptitle('Accuracy by Round - ' + dir, fontsize=24, y=1.0)
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy Difference')
    
    # Adding x-ticks
    round_mod = round_to_nearest_5_or_10(len(reference_data['round']) / 10)
    ticks = [round for round in reference_data['round'] if round % round_mod == 0]
    if 1 not in ticks:
        ticks = [1] + ticks
    ax.set_xticks(ticks)

    # Set y-axis limits to be symmetrical around zero with a small buffer
    y_max = max(abs(ax.get_ylim()[0] - 0.02), abs(ax.get_ylim()[1] + 0.02))
    ax.set_ylim(-y_max, y_max)

    # Add a horizontal line at y=0
    ax.axhline(y=0, color='r', linestyle='-')

    # Add labels for each dataset
    ax.text(0.02, 0.98, "Other CSV files higher", transform=ax.transAxes, 
            verticalalignment='top', fontweight='bold')
    ax.text(0.02, 0.02, f"{os.path.basename(reference_file)} higher", transform=ax.transAxes, 
            verticalalignment='bottom', fontweight='bold')

    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True)

    # Add YAML configuration information to the plot
    config_text = (f"Num Server Rounds: {num_server_rounds}\n"
                   f"Local Epochs: {local_epochs}\n"
                   f"Num Supernodes: {num_supernodes}")
    ax.text(0.95, 0.05, config_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='lightgrey'))

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()