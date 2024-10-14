import pandas as pd
import matplotlib.pyplot as plt
import sys
import yaml

def round_to_nearest_5_or_10(n):
    nearest_5 = round(n / 5) * 5
    nearest_10 = round(n / 10) * 10
    if abs(n - nearest_5) <= abs(n - nearest_10):
        return nearest_5
    else:
        return nearest_10

def main():
    if len(sys.argv) != 4:
        print("Usage: python script.py <input_file1> <input_file2> <config_file>")
        print("Example: python plot.py file1.csv file2.csv config.yaml")
        sys.exit(1)

    file1 = sys.argv[1]
    file2 = sys.argv[2]
    config_file = sys.argv[3]

    # Load YAML configuration
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    # Extracting configuration information
    num_server_rounds = config['server_configuration']['num_server_rounds']
    local_epochs = config['server_configuration']['local_epochs']
    num_supernodes = config['server_configuration']['num_supernodes']

    # Load the CSV data into DataFrames
    data1 = pd.read_csv(file1)
    data2 = pd.read_csv(file2)

    # Ensure both DataFrames have the same number of rounds
    min_rounds = min(len(data1), len(data2))
    data1 = data1.head(min_rounds)
    data2 = data2.head(min_rounds)

    # Calculate the difference in accuracy
    accuracy_diff = data2['accuracy'] - data1['accuracy']

    # Create a plot
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the accuracy difference
    ax.plot(data1['round'], accuracy_diff, linestyle='-', label='Accuracy Difference')

    # Adding title and labels
    ax.set_title('Accuracy Difference by Round')
    ax.set_xlabel('Round')
    ax.set_ylabel('Accuracy Difference')
    
    # Adding x-ticks
    round_mod = round_to_nearest_5_or_10(len(data1['round']) / 10)
    ticks = [round for round in data1['round'] if round % round_mod == 0]
    if 1 not in ticks:
        ticks = [1] + ticks
    ax.set_xticks(ticks)

    # Set y-axis limits to be symmetrical around zero with a small buffer
    y_max = max(abs(accuracy_diff.min() - 0.02), abs(accuracy_diff.max() + 0.02))
    ax.set_ylim(-y_max, y_max)

    # Add a horizontal line at y=0
    ax.axhline(y=0, color='r', linestyle='-')

    # Add labels for each dataset
    ax.text(0.02, 0.98, f"{file2} higher", transform=ax.transAxes, 
            verticalalignment='top', fontweight='bold')
    ax.text(0.02, 0.02, f"{file1} higher", transform=ax.transAxes, 
            verticalalignment='bottom', fontweight='bold')

    ax.legend()
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