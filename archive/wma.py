import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys
import numpy as np

def weighted_moving_average(data, window=5):
    """
    Calculate weighted moving average with more recent values having higher weights
    """
    weights = np.exp(np.linspace(-1., 0., window))
    weights /= weights.sum()
    return pd.Series(data).rolling(window=window, center=True).apply(
        lambda x: np.sum(weights[-len(x):] * x) / np.sum(weights[-len(x):])
    )

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

    csv_directory = dir + '*.csv'
    csv_files = glob.glob(csv_directory)

    plt.figure(figsize=(5, 5))
    all_rounds = None
    window_size = 5

    for csv_file in csv_files:
        data = pd.read_csv(csv_file)
        
        if all_rounds is None:
            all_rounds = data['round']
        
        label = csv_file.split('\\')[-1].split('.')[0]
        label = f"{label}".replace("-out","")

        if(label == "fed_cad"):
            label = "DVSAA-AFL"
            color = 'red'
            marker = 's'  # square marker
        if(label == "fed_avg"):
            label = "FedAvg"
            color = 'blue'
            marker = 'D'  # diamond marker
        
        columns_to_plot = [col for col in data.columns if col not in ['count', 'round']]

        for col in columns_to_plot:
            # Apply weighted moving average
            smoothed_data = weighted_moving_average(data[col], window=window_size)
            
            # Plot the smoothed line
            plt.plot(data['round'], 
                    smoothed_data, 
                    linestyle='-', 
                    color=color,
                    label=f"{label}")
            
            # Add markers every 10 rounds
            marker_indices = data.index[data['round'] % 10 == 0]
            plt.plot(data.loc[marker_indices, 'round'],
                    smoothed_data[marker_indices],
                    marker=marker,
                    linestyle='',
                    fillstyle='none',
                    color=color,
                    markersize=6)

    plt.ylim(0, 1.0)
    plt.xlabel('Training Period')
    plt.ylabel('Value of the accuracy')
    
    max_round = max(all_rounds)
    ticks = list(range(0, max_round + 50, 50))
    plt.xticks(ticks)

    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()