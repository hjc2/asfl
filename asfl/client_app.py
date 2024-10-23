"""asfl: A Flower / PyTorch app."""

import random
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
from collections import Counter

import time
from logging import INFO, DEBUG, CRITICAL, WARNING
from flwr.common.logger import log
import flwr.common.logger as flwr_logger
import numpy as np

from asfl.task import (
    Net,
    DEVICE,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
)

### CLIENT FILE

### CLIENTS ARE SPAWNED FROM WHAT IS DEFINED HERE

# Define Flower Client and client_fn
class FlowerClient(NumPyClient):

    def __init__(self, net, trainloader, valloader, local_epochs, node_id, label_variance):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.node_id = node_id
        self.label_variance = label_variance

    # TRAINS THE CLIENT LOCALLY
    def fit(self, parameters, config):
        set_weights(self.net, parameters)
        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            DEVICE,
        )

        loss, accuracy = test(self.net, self.valloader)

        return get_weights(self.net), len(self.trainloader.dataset), {"loss": loss, "accuracy": accuracy, "labels": self.label_variance}
    
    # RETURNS THE TEST RESULTS AND ACCURACY
    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)

        num_samples = len(self.valloader.dataset)
        
        return loss, num_samples, {"loss": loss, "accuracy": accuracy}

### DEFINES AND SPAWNS THE CLIENTS INITIALLY
def client_fn(context: Context):
    net = Net().to(DEVICE)

    node_id = context.node_id
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader, labels = load_data(context.run_config, partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]


    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs, node_id, labels).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
