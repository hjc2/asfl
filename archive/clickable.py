import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_file>")
        print("Example: python plot.py v1/")
        print("Error: Invalid number of arguments")
        sys.exit(1)
    if '/' not in sys.argv[1]:
        print("Usage: python script.py <input_file>")
        print("Example: python plot.py v1/")
        print("Error: Must be a directory")
        sys.exit(1)

    dir = sys.argv[1]

    plt.style.use('fast')

    # Directory containing your CSV files
    csv_directory = dir + '*.csv'

    # Get a list of all CSV files in the directory
    csv_files = glob.glob(csv_directory)

    # Create a figure and axis object
    fig, ax = plt.subplots(figsize=(10, 6))

    # Initialize a variable to hold the round numbers
    all_rounds = None
    
    # Dictionary to store line objects
    lines = {}

    # Loop through each CSV file
    for csv_file in csv_files:
        # Load the CSV data into a DataFrame
        data = pd.read_csv(csv_file)
        
        # Store round numbers for setting x-ticks later
        if all_rounds is None:
            all_rounds = data['round']
        
        # Determine the label for the plot
        label = csv_file.split('\\')[-1].split('.')[0]
        label = f"{label}".replace("-out","")

        # Define colors and labels
        color_map = {
            "fed_cad": ("DVSAA-AFL", 'red'),
            "fed_avg": ("FedAvg", 'blue'),
            "fed_freq": ("FedFreq", 'green'),
            "fed_equal": ("FedEqual", 'purple'),
            "fed_adaptive": ("FedAdaptive", 'orange')
        }
        
        if label in color_map:
            display_label, color = color_map[label]
        else:
            continue
        
        # Get the columns to plot, excluding 'count' and 'round'
        columns_to_plot = [col for col in data.columns if col not in ['count', 'round']]
        
        # Plot each relevant column
        for col in columns_to_plot:
            line = ax.plot(data['round'], data[col], marker='', linestyle='-', 
                         label=display_label, color=color)[0]
            lines[display_label] = line

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
        legline.set_picker(True)  # Enable picking on the legend line
        legline.set_pickradius(10)  # Make the pickup radius larger
        legline.set_linewidth(3)  # Make the legend lines thicker
        lined[legline] = origline


    # Define the click event handler
    def on_pick(event):
        # On the pick event, get the legend line that was picked
        legline = event.artist
        # Get the original line associated with this legend line
        origline = lined[legline]
        # Get the visibility state
        vis = not origline.get_visible()
        # Set the visibility
        origline.set_visible(vis)
        # Change the alpha on the legend line
        legline.set_alpha(1.0 if vis else 0.2)
        fig.canvas.draw()

    # Connect the pick event to the handler
    fig.canvas.mpl_connect('pick_event', on_pick)
    
    ax.grid(True, color='0.9')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()