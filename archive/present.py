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

    # Directory containing your CSV files (change this as needed)
    csv_directory = dir + '*.csv'  # Use a wildcard to match all CSV files

    # Get a list of all CSV files in the directory
    csv_files = glob.glob(csv_directory)

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Initialize a variable to hold the round numbers
    all_rounds = None

    # Loop through each CSV file
    for csv_file in csv_files:
        # Load the CSV data into a DataFrame
        data = pd.read_csv(csv_file)
        
        # Store round numbers for setting x-ticks later
        if all_rounds is None:
            all_rounds = data['round']
        
        # Determine the label for the plot (extract from filename)
        label = csv_file.split('\\')[-1].split('.')[0]  # Gets the filename without the path and extension
        label = f"{label}".replace("-out","")

        if(label == "fed_cad"):
            label = "DVSAA-AFL"
            color = 'red'
        if(label == "fed_avg"):
            label = "FedAvg"
            color = 'blue'
        
        # Get the columns to plot, excluding 'count' and 'round'
        columns_to_plot = [col for col in data.columns if col not in ['count', 'round']]
        # columns_to_plot = [col for col in data.columns if col not in ['accuracy', 'round']]
        

        # Plot each relevant column
        for col in columns_to_plot:
            # data = data.iloc[::5].copy()
            plt.plot(data['round'], data[col], marker='', linestyle='-', label=f"{label}", color=color)

    plt.ylim(0, 1.0)
    # plt.ylim(0, 100.0)


    plt.xlabel('Training Period')
    plt.ylabel('Value of the accuracy')
    
    # Generate ticks every 50 rounds up to the maximum round
    max_round = max(all_rounds)
    ticks = list(range(0, max_round + 50, 50))
    plt.xticks(ticks)

    plt.legend()
    plt.grid(True)

    plt.show()

if __name__ == "__main__":
    main()