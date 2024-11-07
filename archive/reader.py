import pandas as pd

# Read the CSV data
df = pd.read_csv('v34/fed_avg-out.csv')

# Calculate the sum of the 'count' column
total_count = df['count'].sum()

print(f"Total count: {total_count}")