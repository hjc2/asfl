"""asfl: A Flower / PyTorch app."""

### THIS FILE DOES THE AI STUFF THAT ENABLES ME TO TRAIN A CIFAR-10 DATASET

### HANDLES ALL TRAINING OF THE AI MODEL

### CONTAINS LOTS OF HELPFUL PYTORCH STUFF
### DO NOT TOUCH UNLESS YOU KNOW WHAT YOU ARE DOING

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner, ExponentialPartitioner, DirichletPartitioner, ShardPartitioner
from flwr.common.logger import log
from flwr.common import Context

from logging import CRITICAL


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


fds = None  # Cache FederatedDataset


def load_data(node_config, partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data and calculate label variance.
    
    Returns:
        tuple: (trainloader, testloader, label_variance)
        where label_variance is the variance of label distribution in the training set
    """
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        if node_config["partition"] == "dirichlet":
            partitioner = DirichletPartitioner(
                num_partitions=num_partitions, 
                partition_by="label", 
                alpha=0.5, 
                min_partition_size=10,
                self_balancing=True, 
                seed=42
            )

            fds = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={"train": partitioner},
            )

        elif node_config["partition"] == "iid":
            partitioner = IidPartitioner(num_partitions=num_partitions)
            fds = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={"train": partitioner},
            )
        elif node_config["partition"] == "shard":
            partitioner = ShardPartitioner(
                num_partitions=num_partitions,
                partition_by="label",
                seed=42,
                num_shards_per_partition=2,
            )
            fds = FederatedDataset(
                dataset="uoft-cs/cifar10",
                partitioners={"train": partitioner},
            )
        else:
            raise ValueError("Invalid partitioner: " + node_config["partition"] + " : " + str(node_config["partition"] == "dirichlet"))

    partition = fds.load_partition(partition_id)

    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)

    # Calculate label variance for the training set
    train_labels = partition_train_test["train"]["label"]
    
    # Count frequency of each label
    label_counts = {}
    total_samples = len(train_labels)
    
    for label in train_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Convert counts to proportions
    label_proportions = {k: v/total_samples for k, v in label_counts.items()}
    
    # Calculate variance
    mean_proportion = 1.0 / len(label_counts)  # Expected mean for balanced distribution
    label_variance = sum((p - mean_proportion) ** 2 for p in label_proportions.values()) / len(label_counts)

    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["img"] = [pytorch_transforms(img) for img in batch["img"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    
    return trainloader, testloader, label_variance

def train(net, trainloader, valloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    net.train()
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["img"]
            labels = batch["label"]
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()

    train_loss, train_acc = test(net, trainloader)
    val_loss, val_acc = test(net, valloader)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


def test(net, testloader):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            images = batch["img"].to(DEVICE)
            labels = batch["label"].to(DEVICE)
            outputs = net(images)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy



def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)