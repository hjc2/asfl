import pandas as pd
import matplotlib.pyplot as plt
import glob

# Directory containing your CSV files (change this as needed)
csv_directory = 'v1/*.csv'  # Use a wildcard to match all CSV files

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
    label = csv_file.split('/')[-1].split('.')[0]  # Gets the filename without the path and extension
    
    # Get the columns to plot, excluding 'count' and 'round'
    columns_to_plot = [col for col in data.columns if col not in ['count', 'round']]
    
    # Plot each relevant column
    for col in columns_to_plot:
        plt.plot(data['round'], data[col], marker='o', linestyle='-', label=f"{label} - {col}")

# Adding title and labels after all plots
plt.title('Accuracy by Round')
plt.xlabel('Round')
plt.ylabel('Accuracy')
plt.xticks(all_rounds)  # Set x-ticks to be the round numbers
plt.legend()  # Show legend
plt.grid()
plt.show()
