"""asfl: A Flower / PyTorch app."""

import random
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import time
from logging import INFO, DEBUG, CRITICAL
from flwr.common.logger import log
import flwr.common.logger as flwr_logger

from asfl.task import (
    Net,
    DEVICE,
    load_data,
    get_weights,
    set_weights,
    train,
    test,
)


# Define Flower Client and client_fn
class FlowerClient(NumPyClient):

    def __init__(self, net, trainloader, valloader, local_epochs, node_id):

        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.node_id = node_id

    def fit(self, parameters, config):

        set_weights(self.net, parameters)

        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            DEVICE,
        )

        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        # flwr_logger.configure(identifier="cl: " + str(self.node_id) + " - ", filename="log.txt")
        # log(CRITICAL, f"Node ID: {self.node_id}, Evaluation results - Loss: {loss}, Accuracy: {accuracy}")
        
        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    # log(CRITICAL, str(context.node_id))
    # log(CRITICAL, str(context.node_config ))
    # log(CRITICAL, str(context.run_config))

    net = Net().to(DEVICE)

    node_id = context.node_id
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs, node_id).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
