
# THE CUSTOM STRATEGY FOR ADDING FUNCTIONALITY SUCH AS
# - multiples in the network
# - accuracy logging, aggregation evaluation

# INHERIT FROM THIS STRATEGY, DO NOT USE IT

from flwr.server.strategy import FedAvg
from ..poisson import vehicles_in_round
from typing import Union, Callable, Dict, List, Optional, Tuple
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    NDArrays,
    MetricsAggregationFn,
    Parameters,
    Scalar,
    parameters_to_ndarrays,
)

from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.criterion import Criterion

from flwr.common.logger import log
from logging import WARNING, INFO, DEBUG, CRITICAL, ERROR
from .dat import advlog
import random

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
        return client.cid in self.includeList

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
        cid_ll: List[Tuple[int, List[int]]] = [],
        adv_log: bool = False,
        fraction: int = 2,
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
        self.num_rounds = num_rounds # for poisson
        self.cid_ll = cid_ll # tracks the rounds and the clients selected
                            # used for tracking how long since it was included
        self.good_cid_list = []
        self.adv_log = adv_log
        self.fraction = fraction


    def configure_fit(
        self, server_round: int, parameters: Parameters, client_manager: ClientManager
    ) -> List[Tuple[ClientProxy, FitIns]]:
        """Configure the next round of training."""
        config = {}

        # log(WARNING, "fraction: " + str(self.fraction))        

        if self.on_fit_config_fn is not None:
            # Custom fit config function provided
            config = self.on_fit_config_fn(server_round)
        

        # log(CRITICAL, config.)
        fit_ins = FitIns(parameters, config)

        # Sample clients
        _, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )

        clients = client_manager.all()
        
        advlog(self.adv_log, lambda: log(CRITICAL, "total num of rounds " + str(self.num_rounds)))
        CID_LIST = []

        for x in clients:
            CID_LIST.append(x)

        advlog(self.adv_log, lambda: log(ERROR, "CID_LIST " + str(CID_LIST)))
        random.seed = server_round

        advlog(self.adv_log, lambda: log(CRITICAL, "CID_LIST LEN " + str(len(CID_LIST))))
        advlog(self.adv_log, lambda: log(CRITICAL, "vehicles in round: " + str(vehicles_in_round(self.num_rounds, len(clients), server_round, fraction=self.fraction))))

        self.good_cid_list = random.sample(CID_LIST, vehicles_in_round(self.num_rounds, len(clients), server_round, fraction=self.fraction))
        
        if(self.cid_ll == [] and server_round == 1):
            self.cid_ll.append((0, CID_LIST))
            
        self.cid_ll.append((server_round, self.good_cid_list))

        sample_size = len(self.good_cid_list)

        advlog(self.adv_log, lambda: log(ERROR, "FIT: GOOD CID LIST" + str(self.good_cid_list)))

        advlog(self.adv_log, lambda: log(ERROR, "sample size " + str(sample_size)))
        
        custom = CustomCriterion(self.good_cid_list)

        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=min_num_clients,
            criterion=custom, # Pass custom criterion here
        )

        # Return client/config pairs
        return [(client, fit_ins) for client in clients]
    
    def configure_evaluate(
            self, server_round: int, parameters: Parameters, client_manager: ClientManager
        ) -> List[Tuple[ClientProxy, EvaluateIns]]:
        """Configure the next round of evaluation."""
        # Do not configure federated evaluation if fraction eval is 0.
        if self.fraction_evaluate == 0.0:
            return []

        # Parameters and config
        config = {}
        if self.on_evaluate_config_fn is not None:
            # Custom evaluation config function provided
            config = self.on_evaluate_config_fn(server_round)
        evaluate_ins = EvaluateIns(parameters, config)

        custom = CustomCriterion(self.good_cid_list)

        _, min_num_clients = self.num_fit_clients(
            client_manager.num_available()
        )
        advlog(self.adv_log, lambda: log(ERROR, f"EVAL: good_cid_list {self.good_cid_list}"))
        clients = client_manager.sample(
            num_clients=len(self.good_cid_list),
            min_num_clients=min_num_clients,
            criterion=custom, # Pass custom criterion here
        )

        # Return client/config pairs
        return [(client, evaluate_ins) for client in clients]

    # returns the accuracy and count, etc
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}

        def weighted_avg(results: List[Tuple[int, float]]) -> float:
            """Aggregate evaluation results obtained from multiple clients."""
            num_total_evaluation_examples = sum(num_examples for (num_examples, _) in results)
            weighted_losses = [num_examples * loss for num_examples, loss in results]
            return sum(weighted_losses) / num_total_evaluation_examples

        """Aggregate evaluation losses using weighted average."""
        if not results:
            return None, {}
        # Do not aggregate if there are failures and failures are not accepted
        if not self.accept_failures and failures:
            return None, {}

        # Aggregate loss
        loss_aggregated = weighted_avg(
            [
                (evaluate_res.num_examples, evaluate_res.loss)
                for _, evaluate_res in results
            ]
        )
        metrics_aggregated = {}
            
        # ACCURACY CALCULATIONS
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]
        aggregated_accuracy = sum(accuracies) / sum(examples)

        metrics_aggregated["accuracy"] = aggregated_accuracy
        metrics_aggregated["count"] = len(results)
        # metrics_aggregated["all-acc"] = results

        log(INFO, "aggregated accuracy: " + str(aggregated_accuracy))

        return loss_aggregated, metrics_aggregated