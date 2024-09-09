"""asfl: A Flower / PyTorch app."""

import random
from flwr.client import NumPyClient, ClientApp
from flwr.common import Context
import time
from logging import INFO, DEBUG
from flwr.common.logger import log

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
    def __init__(self, net, trainloader, valloader, local_epochs):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs

    def fit(self, parameters, config):

        set_weights(self.net, parameters)


        results = train(
            self.net,
            self.trainloader,
            self.valloader,
            self.local_epochs,
            DEVICE,
        )

        # # Simulate dropout with a 40% chance
        # randChance = random.random()
        # print(randChance)
        # if random.random() < 0.5:
        #     print("Client dropping out for this round")
        #     time.sleep(config.get("round_duration", 10))  # Simulate dropout duration
        #     return None  # Indicate dropout
        
        return get_weights(self.net), len(self.trainloader.dataset), results

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, accuracy = test(self.net, self.valloader)
        log(INFO, f"Evaluation results - Loss: {loss}, Accuracy: {accuracy}")

        return loss, len(self.valloader.dataset), {"accuracy": accuracy}


def client_fn(context: Context):
    # Load model and data
    net = Net().to(DEVICE)
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions)
    local_epochs = context.run_config["local-epochs"]

    # Return Client instance
    return FlowerClient(net, trainloader, valloader, local_epochs).to_client()


# Flower ClientApp
app = ClientApp(
    client_fn,
)
