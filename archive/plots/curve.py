import pandas as pd
import matplotlib.pyplot as plt
import glob
import yaml
import sys
import numpy as np
from scipy import interpolate

def round_to_nearest_5_or_10(n):
    nearest_5 = round(n / 5) * 5
    nearest_10 = round(n / 10) * 10
    return nearest_5 if abs(n - nearest_5) <= abs(n - nearest_10) else nearest_10

def create_spline(x, y, smoothness=0):
    x_new = np.linspace(x.min(), x.max(), 300)
    spl = interpolate.make_interp_spline(x, y, k=3, bc_type='natural')
    y_smooth = spl(x_new)
    return x_new, y_smooth

def main():
    if len(sys.argv) != 2 or '/' not in sys.argv[1]:
        print("Usage: python script.py <input_directory>")
        print("Example: python plot.py v1/")
        sys.exit(1)

    dir = sys.argv[1]

    plt.style.use('bmh')

    # Load YAML configuration
    with open(dir + 'i.yaml', 'r') as file:
        config = yaml.safe_load(file)

    num_server_rounds = config['server_configuration']['num_server_rounds']
    local_epochs = config['server_configuration']['local_epochs']
    num_supernodes = config['server_configuration']['num_supernodes']
    partitioner = config['server_configuration']['partition']
    fraction = config['server_configuration']['fraction']

    csv_directory = dir + '*.csv'
    csv_files = glob.glob(csv_directory)

    plt.figure(figsize=(12, 8))
    lines = []  # To store line objects for toggling

    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        
        label = csv_file.split('/')[-1].split('.')[0]
        
        columns_to_plot = [col for col in data.columns if col not in ['count', 'round']]
        
        for col in columns_to_plot:
            x = data['round']
            y = data[col]

            # Spline interpolation
            x_smooth, y_smooth = create_spline(x, y)
            line, = plt.plot(x_smooth, y_smooth, linestyle='-', label=f"{label} - {col} (Spline)")
            lines.append(line)  # Add line to the list

    plt.title(f"Partitioner: {partitioner}", fontsize=16)
    plt.suptitle('Accuracy by Round - ' + dir, fontsize=24, y=1.0)
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    
    round_mod = round_to_nearest_5_or_10(len(data['round']) / 10)
    ticks = [round for round in data['round'] if round % round_mod == 0]
    if 1 not in ticks:
        ticks = [1] + ticks
    plt.xticks(ticks)

    legend = plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)

    config_text = (f"Num Server Rounds: {num_server_rounds}\n"
                   f"Local Epochs: {local_epochs}\n"
                   f"Num Supernodes: {num_supernodes}\n"
                   f"Fraction: {fraction}")
    plt.text(0.05, 0.95, config_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', edgecolor='black', facecolor='lightgrey'))

    plt.tight_layout()

    # Set up event handling for toggling lines using legend
    def on_legend_click(event):
        # Get the legend item that was clicked
        for i, line in enumerate(lines):
            if event.artist == legend.get_lines()[i]:  # Compare clicked artist with legend items
                line.set_visible(not line.get_visible())
                legend.get_lines()[i].set_alpha(1.0 if line.get_visible() else 0.2)  # Change legend alpha
        plt.draw()

    fig = plt.gcf()
    fig.canvas.mpl_connect('pick_event', on_legend_click)

    plt.show()

if __name__ == "__main__":
    main()
