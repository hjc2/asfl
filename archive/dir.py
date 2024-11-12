import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys
import os
import yaml
import hashlib
import colorsys

def get_consistent_color(name):
    """Generate a consistent color based on string hash"""
    # Generate hash of the name
    hash_obj = hashlib.md5(name.encode())
    hash_int = int(hash_obj.hexdigest(), 16)
    
    # Use hash to generate HSV values
    # Hue: Use full hash for color variety (0-1)
    hue = (hash_int % 1000) / 1000.0
    # Saturation: Keep relatively high for vibrant colors (0.6-0.9)
    saturation = 0.7 + (hash_int % 200) / 1000.0
    # Value: Keep high for visibility (0.8-1.0)
    value = 0.8 + (hash_int % 200) / 1000.0
    
    # Convert HSV to RGB
    rgb = colorsys.hsv_to_rgb(hue, saturation, value)
    return rgb

def get_display_name(label, epochs):
    """Generate consistent display names for algorithms"""
    name_map = {
        "fed_avg": "FedAvg",
        "fed_final": "DVSAA-AFL",
        "fed_ftrim": "DVSAA-AFL (trim)",
        # Add more mappings as needed
    }
    base_name = name_map.get(label, label)
    return f"{base_name} (E={epochs})"

def main():
    if len(sys.argv) != 3:
        print("Usage: python script.py <base_directory> <filename>")
        print("Example: python plot.py experiments/ accuracy.csv")
        print("Error: Invalid number of arguments")
        sys.exit(1)

    base_dir = sys.argv[1]
    target_file = sys.argv[2]

    if not os.path.isdir(base_dir):
        print("Error: First argument must be a directory")
        sys.exit(1)

    plt.style.use('fast')

    # Find all subdirectories
    subdirs = [d for d in glob.glob(os.path.join(base_dir, "*/")) if os.path.isdir(d)]
    
    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(6, 6))

    # Initialize a variable to hold the round numbers
    all_rounds = None
    
    # Dictionary to store line objects
    lines = {}

    # Loop through each subdirectory
    for subdir in subdirs:
        # Look for the target file in this subdirectory
        target_path = os.path.join(subdir, target_file)
        yaml_path = os.path.join(subdir, "i.yaml")
        
        if not os.path.exists(target_path):
            print(f"Warning: {target_file} not found in {subdir}")
            continue

        if not os.path.exists(yaml_path):
            print(f"Warning: config.yaml not found in {subdir}")
            continue

        # Read the YAML file
        try:
            with open(yaml_path, 'r') as yaml_file:
                config = yaml.safe_load(yaml_file)
                epochs = config.get('server_configuration', {}).get('local_epochs', 'unknown')
        except Exception as e:
            print(f"Warning: Error reading YAML in {subdir}: {e}")
            epochs = 'unknown'

        # Load the CSV data into a DataFrame
        data = pd.read_csv(target_path)
        
        # Store round numbers for setting x-ticks later
        if all_rounds is None:
            all_rounds = data['round']
        
        # Determine the label for the plot (use subdirectory name)
        label = os.path.basename(os.path.dirname(subdir))
        
        # Generate display name and color
        display_label = get_display_name(label, epochs)
        color = get_consistent_color(label)  # Generate color based on label
        
        # Get the columns to plot, excluding 'count' and 'round'
        columns_to_plot = [col for col in data.columns if col not in ['count', 'round']]
        
        # Plot each relevant column
        for col in columns_to_plot:
            line = ax.plot(data['round'], data[col], marker='', linestyle='-', 
                         label=display_label, color=color)[0]
            lines[display_label] = line

    if not lines:
        print("Error: No valid data found to plot")
        sys.exit(1)

    ax.set_ylim(0, 1.0)
    ax.set_xlabel('Training Period')
    ax.set_ylabel('Value of the accuracy')
    
    # Generate ticks every 50 rounds
    max_round = max(all_rounds)
    ticks = list(range(0, max_round + 50, 50))
    ax.set_xticks(ticks)

    # Set ticks to appear on both sides
    ax.yaxis.set_ticks_position('both')
    ax.xaxis.set_ticks_position('both')
    ax.tick_params(axis='y', direction='in')
    ax.tick_params(axis='x', direction='in')
    
    # Create the legend
    leg = ax.legend()
    
    # Add click functionality to legend
    lined = dict()
    for legline, origline in zip(leg.get_lines(), lines.values()):
        legline.set_picker(True)
        legline.set_pickradius(3)
        legline.set_linewidth(3)
        lined[legline] = origline

    def on_pick(event):
        legline = event.artist
        origline = lined[legline]
        vis = not origline.get_visible()
        origline.set_visible(vis)
        legline.set_alpha(1.0 if vis else 0.2)
        fig.canvas.draw()

    fig.canvas.mpl_connect('pick_event', on_pick)
    
    ax.grid(True, color='0.9')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()