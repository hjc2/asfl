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
from flwr_datasets.partitioner import IidPartitioner, ExponentialPartitioner, DirichletPartitioner
from flwr.common.logger import log
from flwr.common import Context

from logging import CRITICAL


from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms import (
    Compose, Normalize, ToTensor, RandomHorizontalFlip,
    RandomCrop, RandomRotation
)

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
fds = None  # Cache FederatedDataset

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Net(nn.Module):
    def __init__(self, num_classes=10):
        super(Net, self).__init__()
        self.in_channels = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.layer1 = self.make_layer(64, 2, stride=1)
        self.layer2 = self.make_layer(128, 2, stride=2)
        self.layer3 = self.make_layer(256, 2, stride=2)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)

    def make_layer(self, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(ResidualBlock(self.in_channels, out_channels, stride))
            self.in_channels = out_channels
        return nn.Sequential(*layers)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

def load_data(node_config, partition_id: int, num_partitions: int):
    """Load partition CIFAR10 data with augmentation."""
    # Create partitioner for each call
    if node_config["partition"] == "dirichlet":
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions,
            partition_by="label",
            alpha=0.5,
            min_partition_size=10,
            self_balancing=True,
            seed=42
        )
    elif node_config["partition"] == "iid":
        partitioner = IidPartitioner(num_partitions=num_partitions)
    else:
        raise ValueError(f"Invalid partitioner: {node_config['partition']}")
        
    # Create new FederatedDataset each time
    fds = FederatedDataset(
        dataset="uoft-cs/cifar10",
        partitioners={"train": partitioner},
    )

    partition = fds.load_partition(partition_id)
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    
    # Calculate label variance
    train_labels = partition_train_test["train"]["label"]
    label_counts = {}
    total_samples = len(train_labels)
    
    for label in train_labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    label_proportions = {k: v/total_samples for k, v in label_counts.items()}
    mean_proportion = 1.0 / len(label_counts)
    label_variance = sum((p - mean_proportion) ** 2 for p in label_proportions.values()) / len(label_counts)

    # Enhanced data augmentation for training
    train_transforms = Compose([
        ToTensor(),
        RandomCrop(32, padding=4),
        RandomHorizontalFlip(),
        RandomRotation(15),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    # Simpler transforms for testing
    test_transforms = Compose([
        ToTensor(),
        Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    def apply_train_transforms(batch):
        batch["img"] = [train_transforms(img) for img in batch["img"]]
        return batch

    def apply_test_transforms(batch):
        batch["img"] = [test_transforms(img) for img in batch["img"]]
        return batch

    train_data = partition_train_test["train"].with_transform(apply_train_transforms)
    test_data = partition_train_test["test"].with_transform(apply_test_transforms)
    
    # Reduced batch size and removed num_workers/pin_memory
    trainloader = DataLoader(train_data, batch_size=32, shuffle=True)
    testloader = DataLoader(test_data, batch_size=32)
    
    return trainloader, testloader, label_variance

def train(net, trainloader, valloader, epochs, device):
    """Train the model with improved training process but lower memory usage."""
    net.to(device)
    criterion = nn.CrossEntropyLoss().to(device)
    
    optimizer = torch.optim.AdamW(net.parameters(), lr=0.001, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        net.train()
        
        for batch in trainloader:
            images = batch["img"].to(device)
            labels = batch["label"].to(device)
            
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=1.0)
            optimizer.step()
            
            # Clear some memory
            del images, labels, outputs, loss
            
        scheduler.step()
        
        # Validation phase
        val_loss, val_acc = test(net, valloader)
        
        # Only save accuracy value, not the whole model
        if val_acc > best_val_acc:
            best_val_acc = val_acc

    train_loss, train_acc = test(net, trainloader)
    val_loss, val_acc = test(net, valloader)

    return {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }

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