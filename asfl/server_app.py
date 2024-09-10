"""asfl: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from asfl.task import Net, get_weights
from flwr.server.strategy import Strategy

from .dvsaa_afl import DVSAAAFL
from .dvsaa_afl import FedCustom

from typing import Union

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from typing import Dict, List, Optional, Tuple


# Initialize model parameters
ndarrays = get_weights(Net())
parameters = ndarrays_to_parameters(ndarrays)


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    strat_mode = context.run_config["strat-mode"]

    # Define strategy

    # write something that maps strings to the strat i want
    if strat_mode == 'dvsaa':
        strategy = DVSAAAFL(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
            # initial_parameters=parameters,
        )
    elif strat_mode == 'fedcustom':
        strategy = FedCustom(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            # max_rounds=num_rounds,
        )
    elif strat_mode == 'fedavg':
        strategy = FedAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
        )

    print("running in " + strat_mode)
    
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

def client_manager_fn(context: Context):
    return None

# Create ServerApp
app = ServerApp(server_fn=server_fn)
