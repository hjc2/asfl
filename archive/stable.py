import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys

def analyze_accuracy_steps(data, label):
    # Calculate step differences
    data['accuracy_diff'] = data['accuracy'].diff().abs()
    
    # Calculate statistics
    stats_dict = {
        'algorithm': label,
        'mean_diff': data['accuracy_diff'].mean(),
        'median_diff': data['accuracy_diff'].median(),
        'std_diff': data['accuracy_diff'].std(),
        'max_improvement': data['accuracy_diff'].max(),
        'worst_decline': data['accuracy_diff'].min(),
        'positive_steps': (data['accuracy_diff'] > 0).sum(),
        'negative_steps': (data['accuracy_diff'] < 0).sum(),
        'total_improvement': data['accuracy'].iloc[-1] - data['accuracy'].iloc[0],
        'final_accuracy': data['accuracy'].iloc[-1]
    }
    
    return stats_dict, data['accuracy_diff']

def main():
    if len(sys.argv) != 2:
        print("Usage: python script.py <input_directory>")
        print("Example: python analyze.py v1/")
        print("Error: Invalid number of arguments")
        sys.exit(1)
    if '/' not in sys.argv[1]:
        print("Usage: python script.py <input_directory>")
        print("Example: python analyze.py v1/")
        print("Error: Must be a directory")
        sys.exit(1)

    dir = sys.argv[1]
    csv_directory = dir + '*.csv'
    csv_files = glob.glob(csv_directory)

    # Create figures
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(5, 6))
    
    # Store results for comparison
    all_stats = []
    
    # Process each file
    for csv_file in csv_files:
        # Extract label from filename
        label = csv_file.split('\\')[-1].split('.')[0]
        label = label.replace("-out", "")
        
        if label == "fed_cad":
            label = "DVSAA-AFL"
            color = 'red'
        if label == "fed_avg":
            label = "FedAvg"
            color = 'blue'
        
        # Read and analyze data
        data = pd.read_csv(csv_file)
        stats, diffs = analyze_accuracy_steps(data, label)
        all_stats.append(stats)
        
        # Plot accuracy differences
        ax1.plot(data.index, diffs, label=f'{label}', color=color, alpha=0.7)
        ax2.plot(data.index, data['accuracy'], label=f'{label}', color=color)

    # Configure difference plot
    ax1.axhline(y=0, color='k', linestyle='--', alpha=0.3)
    ax1.set_title('Accuracy Differences Between Rounds')
    ax1.set_xlabel('Round')
    ax1.set_ylabel('Difference')
    ax1.legend()
    ax1.grid(True, color='0.9')
    ax1.tick_params(axis='both', direction='in')
    
    # Configure accuracy plot
    ax2.set_title('Overall Accuracy Progression')
    ax2.set_xlabel('Round')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, color='0.9')
    ax2.tick_params(axis='both', direction='in')
    
    plt.tight_layout()
    
    # Print comparative statistics
    print("\nComparative Analysis:")
    stats_df = pd.DataFrame(all_stats)
    stats_df = stats_df.set_index('algorithm')
    
    print("\nKey Statistics:")
    print(stats_df[['mean_diff', 'std_diff', 'total_improvement', 'final_accuracy']].round(4))
    
    print("\nStep Counts:")
    print(stats_df[['positive_steps', 'negative_steps']].astype(int))
    
    print("\nExtreme Changes:")
    print(stats_df[['max_improvement', 'worst_decline']].round(4))
    
    plt.show()

if __name__ == "__main__":
    main()