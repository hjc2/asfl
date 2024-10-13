import pandas as pd
import matplotlib.pyplot as plt
import glob

# Directory containing your CSV files (change this as needed)
csv_directory = 'v1/*.csv'  # Use a wildcard to match all CSV files

# Get a list of all CSV files in the directory
csv_files = glob.glob(csv_directory)

# Create a plot
plt.figure(figsize=(10, 6))

# Loop through each CSV file
for csv_file in csv_files:
    # Load the CSV data into a DataFrame
    data = pd.read_csv(csv_file)
    
    # Determine the label for the plot (extract from filename)
    label = csv_file.split('/')[-1].split('.')[0]  # Gets the filename without the path and extension
    
    # Check for the presence of the correct column and plot accordingly
    if 'FedAcc' in data.columns:
        plt.plot(data['round'], data['FedAcc'], marker='o', linestyle='-', label=label)
    elif 'FedAvg' in data.columns:
        plt.plot(data['round'], data['FedAvg'], marker='o', linestyle='-', label=label)
    else:
        print(f"Warning: Neither 'FedAcc' nor 'FedAvg' found in {csv_file}")

# Adding title and labels
plt.title('Accuracy by Round')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.xticks(data['round'])  # Set x-ticks to be the round numbers
plt.legend()  # Show legend
plt.grid()
plt.show()
