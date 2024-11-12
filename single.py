from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    IidPartitioner,
    DirichletPartitioner,
    ShardPartitioner,
)
import matplotlib.pyplot as plt
import numpy as np

# Setup
num_partitions = 50
partitioner_list = []
title_list = ["IID Partitioning", "Dirichlet Partitioning", "Shard Partitioning"]

# Create partitioners
# IID
fds = FederatedDataset(
    dataset="cifar10",
    partitioners={
        "train": IidPartitioner(num_partitions=num_partitions)
    }
)
partitioner_list.append(fds.partitioners["train"])

# Dirichlet
fds = FederatedDataset(
    dataset="cifar10",
    partitioners={
        "train": DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=0.5,
            seed=42,
            min_partition_size=10,
            self_balancing=True,
        ),
    },
)
partitioner_list.append(fds.partitioners["train"])

# Shard
fds = FederatedDataset(
    dataset="cifar10",
    partitioners={
        "train": ShardPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            seed=42,
            num_shards_per_partition=3,
        ),
    },
)
partitioner_list.append(fds.partitioners["train"])

# Create subplots
fig = plt.figure(figsize=(20, 6))
fig.suptitle("Comparison of Different Partitioning Strategies", fontsize=16, y=1.05)

# Plot each partitioning strategy
for idx, (partitioner, title) in enumerate(zip(partitioner_list, title_list)):
    ax = plt.subplot(1, 3, idx + 1)
    
    # Get label distributions
    label_counts = []
    for i in range(num_partitions):
        partition = partitioner.get_partition(i)
        labels = partition['label'].value_counts().sort_index()
        label_counts.append(labels)
    
    # Stack data for plotting
    data = np.vstack([counts.values for counts in label_counts])
    
    # Create stacked bar plot
    bottom = np.zeros(num_partitions)
    for label in range(10):  # CIFAR-10 has 10 classes
        values = data[:, label]
        ax.bar(range(num_partitions), values, bottom=bottom, 
               label=f'Class {label}', width=1)
        bottom += values
    
    # Customize subplot
    ax.set_title(title)
    ax.set_xlabel('Partition ID')
    ax.set_ylabel('Number of Samples')
    ax.set_xlim(-1, num_partitions)
    
    # Only show legend for the last subplot
    if idx == 2:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Adjust layout to prevent overlap
plt.tight_layout()
plt.show()