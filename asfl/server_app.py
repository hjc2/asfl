"""asfl: A Flower / PyTorch app."""

### SPAWNS THE SERVER APP

### CREATES THE SERVER_FN, GRABS FROM THE CONFIGS, RUNS THE SERVER APP

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from asfl.task import Net, get_weights
from flwr.server.strategy import Strategy


# from .strats.dvsaa_afl import FedCustom
from .strats.federal_avg import FederalAvg
from .strats.fed_agg import FedAgg

from typing import Union
from logging import WARNING, INFO, DEBUG, CRITICAL
from flwr.common.logger import log
import flwr.common.logger as flwr_logger

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
    local_epochs = context.run_config["local-epochs"]
    file_writing = context.run_config["file-writing"]
    inplace_setter = context.run_config["inplace"]
    adv_log_setter = context.run_config["adv-logs"]

    # Define strategy

    strategy = None

    if strat_mode == 'fed_agg':
        strategy = FedAgg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            num_rounds=num_rounds,
            inplace=inplace_setter,
            advanced_logging=adv_log_setter
        )
    elif strat_mode == 'fedavg':
        strategy = FederalAvg(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_available_clients=2,
            initial_parameters=parameters,
            num_rounds=num_rounds,
            inplace=inplace_setter,
            advanced_logging=adv_log_setter
        )
    if file_writing:
        flwr_logger.configure(identifier="dv -", filename="log.txt")

    log(CRITICAL, "file writing: " + str(file_writing))
    log(CRITICAL, "running in " + strat_mode)
    log(CRITICAL, "min num clients " + str(2))
    log(CRITICAL, "num server rounds " + str(num_rounds))
    log(CRITICAL, "num local epochs " + str(local_epochs))


    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

def client_manager_fn(context: Context):
    return None

# Create ServerApp
app = ServerApp(server_fn=server_fn)

