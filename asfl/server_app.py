"""asfl: A Flower / PyTorch app."""

### SPAWNS THE SERVER APP

### CREATES THE SERVER_FN, GRABS FROM THE CONFIGS, RUNS THE SERVER APP

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from asfl.task import Net, get_weights

# from .strats.dvsaa_afl import FedCustom
from .strats.federal_avg import FederalAvg
from .strats.fed_agg import FedAgg
from .strats.fed_acc import FedAcc
from .strats.fed_loss import FedLoss
from .strats.fed_fuzz import FedFuzz
from .strats.fed_equal import FedEqual

from typing import Union
from logging import WARNING, INFO, DEBUG, CRITICAL
from flwr.common.logger import log
import flwr.common.logger as flwr_logger

from flwr.common import (
    ndarrays_to_parameters,
)



# Initialize model parameters
ndarrays = get_weights(Net())
parameters = ndarrays_to_parameters(ndarrays)

def create_strategy(strat_mode, parameters, set_num_rounds, inplace_setter, adv_log_setter, fit_config):
    """Factory function to create the appropriate strategy based on the strat_mode."""
    
    strategies = {
        'fedavg': FederalAvg,
        'fed_agg': FedAgg,
        'fed_acc': FedAcc,
        'fed_loss': FedLoss,
        'fed_fuzz': FedFuzz,
        'fed_equal': FedEqual,
    }

    if strat_mode not in strategies:
        raise ValueError(f"Unknown strategy mode: {strat_mode}")

    strategy_class = strategies[strat_mode]
    
    # Create the strategy instance
    return strategy_class(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        num_rounds=set_num_rounds,
        inplace=inplace_setter,
        adv_log=adv_log_setter,
        on_fit_config_fn=fit_config,
    )

def server_fn(context: Context):

    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    strat_mode = context.run_config["strat-mode"]
    local_epochs = context.run_config["local-epochs"]
    file_writing = context.run_config["file-writing"]
    inplace_setter = context.run_config["inplace"]
    adv_log_setter = context.run_config["adv-logs"]
    log_file_path = strat_mode + ".txt"

    # Define strategy

    def fit_config(server_round: int):
        config = {
            "server_round": server_round,  # The current round of federated learning
            "local_epochs": context.run_config["local-epochs"]
        }
        return config

    strategy = create_strategy(strat_mode, parameters, num_rounds, inplace_setter, adv_log_setter, fit_config)

    if file_writing:
        flwr_logger.configure(identifier="dv -", filename=log_file_path)

    log(INFO, "file writing: " + str(file_writing))
    log(INFO, "running in " + strat_mode)
    log(INFO, "min num clients " + str(2))
    log(INFO, "num server rounds " + str(num_rounds))
    log(INFO, "config num local epochs " + str(local_epochs))
    log(INFO, "advanced logging " + str(adv_log_setter))


    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

def client_manager_fn(context: Context):
    return None

# Create ServerApp
app = ServerApp(server_fn=server_fn)

