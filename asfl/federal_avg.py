
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
from asfl.task import Net, get_weights
from flwr.server.strategy import Strategy

import np as np

from .poisson import vehicles_in_round

from .agg import adapt_aggregate_evaluate

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
from logging import WARNING, INFO, DEBUG, CRITICAL, ERROR
import random

from .dvsaa_afl import FedCustom

WARNING_MIN_AVAILABLE_CLIENTS_TOO_LOW = """
Setting `min_available_clients` lower than `min_fit_clients` or
`min_evaluate_clients` can cause the server to fail when there are too few clients
connected to the server. `min_available_clients` must be set to a value larger
than or equal to the values of `min_fit_clients` and `min_evaluate_clients`.
"""
class CustomCriterion(Criterion):
    def __init__(self, includeList: List[int] = []) -> None:
        self.includeList = includeList

    def select(self, client: ClientProxy) -> bool:
        # print("client: ", client)
        # print("cluent.cid: ", client.cid)
        # print("exclude list: ", self.includeList)
        return client.cid in self.includeList

class FederalAvg(FedCustom):
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}

        aggregated_loss, aggregated_metrics = adapt_aggregate_evaluate(self, server_round, results, failures)
    
        # ACCURACY CALCULATIONS
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        aggregated_accuracy = sum(accuracies) / sum(examples)

        # Return information back
        return aggregated_loss, {"accuracy": aggregated_accuracy, "count": len(results)}