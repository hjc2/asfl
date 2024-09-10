
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from asfl.task import Net, get_weights
from flwr.server.strategy import Strategy

import np as np

from .poisson import vehicles_in_round


from typing import Union, Callable, Dict, List, Optional, Tuple

from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    NDArrays,
    MetricsAggregationFn,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy.aggregate import aggregate, weighted_loss_avg
from flwr.common.logger import log
from logging import WARNING, INFO, DEBUG, CRITICAL

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""
class CustomCriterion(Criterion):
    def __init__(self, excludeList: List[int] = []) -> None:
        self.excludeList = excludeList

    def select(self, client: ClientProxy) -> bool:
        print("client: ", client)
        print("cluent.cid: ", client.cid)
        print("exclude list: ", self.excludeList)
        return client.cid not in self.excludeList
    
    # return client.cid not in BAD_CID_LIST
    # return 10

class FedCustom(FedAvg):

    def __init__(
        self,
        *,
        fraction_fit: float = 1.0,
        fraction_evaluate: float = 1.0,
        min_fit_clients: int = 2,
        min_evaluate_clients: int = 2,
        min_available_clients: int = 2,
        evaluate_fn: Optional[
            Callable[
                [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]],
            ]
        ] = None,
        on_fit_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        on_evaluate_config_fn: Optional[Callable[[int], Dict[str, Scalar]]] = None,
        accept_failures: bool = True,
        initial_parameters: Optional[Parameters] = None,
        fit_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        evaluate_metrics_aggregation_fn: Optional[MetricsAggregationFn] = None,
        inplace: bool = True,
        num_rounds: int = 1,
    ) -> None:
        super().__init__()

        if (
            min_fit_clients > min_available_clients
            or min_evaluate_clients > min_available_clients
        ):
            log(WARNING, WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW)

        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.min_fit_clients = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.min_available_clients = min_available_clients
        self.evaluate_fn = evaluate_fn
        self.on_fit_config_fn = on_fit_config_fn
        self.on_evaluate_config_fn = on_evaluate_config_fn
        self.accept_failures = accept_failures
        self.initial_parameters = initial_parameters
        self.fit_metrics_aggregation_fn = fit_metrics_aggregation_fn
        self.evaluate_metrics_aggregation_fn = evaluate_metrics_aggregation_fn
        self.inplace = inplace
        self.num_rounds = num_rounds  

    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}
        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        _, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        log(CRITICAL, "min num clients " + str(min_num_clients))

        clients = client_manager.all()
        
        log(CRITICAL, "clients" + str(clients))
        log(CRITICAL, "clients[0] " + str(list(clients)[0]))
        log(CRITICAL, "total num of rounds " + str(self.num_rounds))
        log(CRITICAL, "next(iter(clients))" + str(next(iter(clients))))
        
        first_key = next(iter(clients))
        first_elem = clients[first_key]


        BAD_CID_LIST = []
        BAD_CID_LIST.append(first_elem.cid)

        sample_size = len(clients) - len(BAD_CID_LIST)

        log(CRITICAL, "sample size " + str(sample_size))
        
        custom = CustomCriterion(BAD_CID_LIST)

        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            criterion=custom, # Pass custom criterion here
        )

        log(CRITICAL, "total num of rounds " + str(self.num_rounds))

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
