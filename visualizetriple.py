from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import DirichletPartitioner
from flwr_datasets.visualization import plot_label_distributions
from flwr_datasets.visualization import plot_comparison_label_distribution
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import (
    IidPartitioner,
    DirichletPartitioner,
    ShardPartitioner,
)
import matplotlib.pyplot as plt

num_partitions = 50

partitioner_list = []
# title_list = ["IidPartitioner", "DirichletPartitioner", "ShardPartitioner"]
title_list = ["IidPartitioner", "DirichletPartitioner", "ShardPartitioner"]

# IID
fds = FederatedDataset(
    dataset="cifar10",
    partitioners={
        "train" : IidPartitioner(num_partitions=num_partitions)
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

fds = FederatedDataset(
    dataset="cifar10",
    partitioners = {
        "train": ShardPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            seed=42,
            num_shards_per_partition=3,
        ),
    },
)
partitioner_list.append(fds.partitioners["train"])

# Shard

fig, axes, df_list = plot_comparison_label_distribution(
    partitioner_list=partitioner_list,
    label_name="label",
    subtitle="Comparison of Partitioning Schemes on CIFAR10",
    titles=title_list,
    legend=True,
    verbose_labels=True,
)
plt.show()
